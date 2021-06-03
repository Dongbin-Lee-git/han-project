package com.example.camx_test


import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
//import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.camx_test.Retrofit.IUploadAPI
import com.example.camx_test.Retrofit.RetrofitClient
//import com.karumi.dexter.Dexter
//import com.karumi.dexter.PermissionToken
//import com.karumi.dexter.listener.PermissionDeniedResponse
//import com.karumi.dexter.listener.PermissionGrantedResponse
//import com.karumi.dexter.listener.PermissionRequest
//import com.karumi.dexter.listener.single.PermissionListener
import kotlinx.android.synthetic.main.activity_main.*
import okhttp3.MediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody
import java.io.File
import java.nio.ByteBuffer
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

typealias LumaListener = (luma: Double) -> Unit

class MainActivity : AppCompatActivity(){


    lateinit var mService: IUploadAPI


    private val apiUpload: IUploadAPI
        get() = RetrofitClient.client.create(IUploadAPI::class.java)

    private var imageCapture: ImageCapture? = null

    private lateinit var outputDirectory: File
    private lateinit var cameraExecutor: ExecutorService

    private var time = 0
    private var timerTask:Timer? = null
    private var goaltime : Int = 40 * 60 //초

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        if (Build.VERSION.SDK_INT < 16) {
            window.setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN)
        }
        else{
            window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_FULLSCREEN
            actionBar?.hide()
        }


        setContentView(R.layout.activity_main)
//        Dexter.withActivity(this)
//            .withPermission(android.Manifest.permission.READ_EXTERNAL_STORAGE)
//            .withListener(object : PermissionListener {
//                override fun onPermissionGranted(response: PermissionGrantedResponse?) {
//
//                }
//
//                override fun onPermissionDenied(response: PermissionDeniedResponse?) {
//                    Toast.makeText(this@MainActivity, "You must accept permission", Toast.LENGTH_LONG).show()
//                }
//
//                override fun onPermissionRationaleShouldBeShown(
//                    permission: PermissionRequest?,
//                    token: PermissionToken?
//                ) {
//
//                }
//
//            }).check()
        mService =apiUpload

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }



        //service


        // Set up the listener for take photo button
        camera_capture_button.setOnClickListener { start() }
        camera_stop_button.setOnClickListener { stop() }
        camera_pause_button.setOnClickListener { pause() }
        camera_restart_button.setOnClickListener { start() }

        outputDirectory = getOutputDirectory()

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun start() {
        imageView.visibility = View.INVISIBLE
        text.visibility = View.VISIBLE
        text2.visibility = View.INVISIBLE
        camera_capture_button.visibility = View.INVISIBLE
        camera_pause_button.visibility = View.VISIBLE
        camera_stop_button.visibility = View.VISIBLE
        camera_restart_button.visibility = View.INVISIBLE

        timerTask = kotlin.concurrent.timer(period = 1000) {
            time++
            val hour : Int = time / 3600 % 24
            val min : Int = time / 60 % 60
            val sec = time

            if(sec % 3 == 0){
                takePhoto()
            }

            if(sec == goaltime){
                stop()
            }

            runOnUiThread {
                text.text = "$hour : $min -> ${(goaltime/3600).toInt()} : ${(goaltime/60).toInt()}"
            }


        }
    }

    private fun stop(){
        timerTask?.cancel();
        time = 0
        imageView.visibility = View.VISIBLE
        text.visibility = View.INVISIBLE
        text2.visibility = View.VISIBLE
        camera_capture_button.visibility = View.VISIBLE
        camera_pause_button.visibility = View.INVISIBLE
        camera_stop_button.visibility = View.INVISIBLE
        camera_restart_button.visibility = View.INVISIBLE
    }

    private fun pause() {
        timerTask?.cancel();
        imageView.visibility = View.INVISIBLE
        text.visibility = View.VISIBLE
        text2.visibility = View.INVISIBLE
        camera_capture_button.visibility = View.INVISIBLE
        camera_pause_button.visibility = View.INVISIBLE
        camera_stop_button.visibility = View.VISIBLE
        camera_restart_button.visibility = View.VISIBLE

    }

    private fun uploadFile(file: File) {
        if(file !=null){
                val surveyBody = RequestBody.create(MediaType.parse("image/*"), file)
                val body = MultipartBody.Part.createFormData("image", file.name, surveyBody)
                mService.uploadFile(body)

        }

    }

//    override fun onRequestPermissionsResult(
//        requestCode: Int, permissions: Array<String>, grantResults:
//        IntArray) {
//        if (requestCode == REQUEST_CODE_PERMISSIONS) {
//            if (allPermissionsGranted()) {
//                startCamera()
//            } else {
//                Toast.makeText(this,
//                    "Permissions not granted by the user.",
//                    Toast.LENGTH_SHORT).show()
//                finish()
//            }
//        }
//    }

    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case

        val imageCapture = imageCapture ?: return

        // Create time-stamped output file to hold the image
        val photoFile = File(
            outputDirectory,
            "capture.jpg"
        )

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions, ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    uploadFile(photoFile)
                    val savedUri = Uri.fromFile(photoFile)
                    val msg = "Photo capture succeeded: $savedUri"
                    //Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d(TAG, msg)
                }
            })
    }


    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview


            imageCapture = ImageCapture.Builder()
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, LuminosityAnalyzer { luma ->
                        Log.d(TAG, "Average luminosity: $luma")
                    })
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, imageCapture, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let {
            File(it, resources.getString(R.string.app_name)).apply { mkdirs() } }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }

    private class LuminosityAnalyzer(private val listener: LumaListener) : ImageAnalysis.Analyzer {

        private fun ByteBuffer.toByteArray(): ByteArray {
            rewind()    // Rewind the buffer to zero
            val data = ByteArray(remaining())
            get(data)   // Copy the buffer into a byte array
            return data // Return the byte array
        }

        override fun analyze(image: ImageProxy) {

            val buffer = image.planes[0].buffer
            val data = buffer.toByteArray()
            val pixels = data.map { it.toInt() and 0xFF }
            val luma = pixels.average()

            listener(luma)

            image.close()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXBasic"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

}