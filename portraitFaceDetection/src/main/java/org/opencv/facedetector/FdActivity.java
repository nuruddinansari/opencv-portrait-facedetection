package org.opencv.facedetector;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class FdActivity extends AppCompatActivity implements CvCameraViewListener2 {

    private static final String TAG = "OCVPApp::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    private static final int MY_PERMISSIONS_REQUEST_CAMERA = 455;

    private Mat mRgba;
    private Mat mGray;
    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;

    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detectionBasedTracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                if (ActivityCompat.shouldShowRequestPermissionRationale((Activity) this, Manifest.permission.CAMERA)) {

                } else {
                    ActivityCompat.requestPermissions((Activity) this, new String[]{Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
                }
            }
        }

    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();

        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = rotateMat(inputFrame.gray());

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();

        if (mJavaDetector != null)
            mJavaDetector.detectMultiScale(mGray, faces, 1.1, 3, 2, new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());

        mGray.release();

        Mat newMat = rotateMat(mRgba);

        for (Rect rect : faces.toArray()) {
            Imgproc.rectangle(newMat, rect.tl(), rect.br(), FACE_RECT_COLOR, 3);

            Imgproc.line(newMat, new Point(rect.tl().x, rect.tl().y), new Point(rect.tl().x, rect.tl().y + 50), new Scalar(0, 255, 255), 3, 8, 0);
            Imgproc.line(newMat, new Point(rect.tl().x, rect.tl().y), new Point(rect.tl().x + 50, rect.tl().y), new Scalar(0, 255, 255), 3, 8, 0);
            Imgproc.line(newMat, new Point(rect.br().x, rect.br().y), new Point(rect.br().x, rect.br().y - 50), new Scalar(0, 255, 255), 3, 8, 0);
            Imgproc.line(newMat, new Point(rect.br().x, rect.br().y), new Point(rect.br().x - 50, rect.br().y), new Scalar(0, 255, 255), 3, 8, 0);

            Imgproc.line(newMat, new Point((rect.br().x - rect.width) + 20, rect.br().y), new Point((rect.br().x - rect.width) + 20, rect.br().y + 60), new Scalar(0, 255, 255), 1, 8, 0);
            Imgproc.line(newMat, new Point((rect.br().x - rect.width) + 20, rect.br().y + 60), new Point((rect.br().x - rect.width) + 60, rect.br().y + 80), new Scalar(0, 255, 255), 1, 8, 0);

            Imgproc.circle(newMat, new Point(rect.br().x - rect.width + 60, rect.br().y + 80), 3, new Scalar(0, 255, 255));


            Imgproc.putText(newMat, "NURUDDIN", new Point((rect.br().x - rect.width) + 70, rect.br().y + 90),
                    Core.FONT_HERSHEY_TRIPLEX, 0.8, new Scalar(0, 0, 0));

        }

        Imgproc.resize(newMat, mRgba, new Size(mRgba.width(), mRgba.height()));
        newMat.release();

        return mRgba;
    }

    public Mat rotateMat(Mat matImage) {
        Mat rotated = matImage.t();
        Core.flip(rotated, rotated, 1);
        return rotated;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST_CAMERA:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(getApplicationContext(), "Permission Granted", Toast.LENGTH_SHORT).show();
                    // main logic
                } else {
                    Toast.makeText(getApplicationContext(), "Permission Denied", Toast.LENGTH_SHORT).show();
                }
                break;
        }
    }
}
