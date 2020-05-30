package com.iot.smart

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.media.Image
import android.net.Uri
import android.os.*
import android.util.AttributeSet
import android.util.Size
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.iot.smart.tf.Classifier
import com.iot.smart.tf.Classifier.Recognition
import com.iot.smart.tf.TFLiteObjectDetectionAPIModel
import com.iot.smart.utils.ImageUtils
import com.iot.smart.utils.SingleBoxTracker
import kotlinx.android.synthetic.main.activity_detection.*
import timber.log.Timber
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.time.ExperimentalTime
import kotlin.time.milliseconds


typealias DetectionListener = (results: Bitmap) -> Unit

class DetectionActivity : AppCompatActivity(){
    // Configuration values for the prepackaged SSD model.
    private val isQuantized = true
    private val apiInputSize = 300
    private val  modelFile = "detect.tflite"
    private val labelFile = "file:///android_asset/labelmap.txt"
    private var detector: Classifier? = null

    private var preview: Preview? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var lastProcessingTimeMs: Long = 0
    private var cropCopyBitmap: Bitmap? = null
    private  var startStreaming:Long = 0
    private val classes = listOf<String>("person")
    private enum class DetectorMode {
        TF_OD_API
    }
    private var cameraMode: Int = 1
    private val modeApi: DetectorMode = DetectorMode.TF_OD_API
    private val confidenceTf= 0.5f
    private lateinit var outputDirectory: File
    private lateinit var cameraExecutor: ExecutorService
    var trackingOverlay: OverlayView? = null

    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null

    private val imageSize = Size(640, 480)
    private var isProcessing = false

    private var timestamp:Long = 0
    private var totalCount:Int = 0

    private var handler: Handler? = null
    private var handlerThread: HandlerThread? = null

    private var tracker: SingleBoxTracker? = null
    private var threadsTextView: TextView? = null
    private var saveShot = false


    @ExperimentalTime
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detection)

        val toolbar: Toolbar = findViewById(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar!!.setDisplayShowTitleEnabled(true)
        supportActionBar!!.setDisplayHomeAsUpEnabled(true)
        supportActionBar!!.setDisplayShowHomeEnabled(true)

        toolbar.setNavigationIcon(R.drawable.ic_action_navigation)
        // Setup the listener for take photo button
        camera_capture_button.setOnClickListener { takePhoto() }
        outputDirectory = getOutputDirectory()
        cameraExecutor = Executors.newSingleThreadExecutor()
        setUp()
        cameraMode = intent.getIntExtra("cameraMode", 0)
            Timber.i("camera mode $cameraMode")
        // Camera mode
        if (cameraMode == 1){
            supportActionBar?.title = "Smart Camera"
            camera_capture_button.visibility = View.VISIBLE
            stats_container.visibility = View.GONE
            cameraPreview.visibility = View.VISIBLE
        }else{
            supportActionBar?.title = "Smart Video"
            camera_capture_button.visibility = View.GONE
            cameraPreview.visibility = View.GONE
            viewFinder.visibility = View.VISIBLE
            stats_container.visibility = View.VISIBLE

        }

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                requestPermissions(REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
            }
            startCamera()
        }

    }
    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }
    @Synchronized
    override fun onResume() {
        Timber.d("onResume $this")
        super.onResume()
        setUp()
        handlerThread = HandlerThread("inference")
        handlerThread!!.start()
        handler = Handler(handlerThread!!.looper)
    }

    @Synchronized
    override fun onPause() {
        Timber.d("onPause $this")
        handlerThread!!.quitSafely()
        try {
            handlerThread!!.join()
            handlerThread = null
            handler = null
        } catch (e: InterruptedException) {
            Timber.e(e, "Exception!")
        }
        super.onPause()
    }
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            // Preview
            preview = Preview.Builder()
                .build()
            // Select back camera
            val cameraSelector = CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()
                // Bind use cases to camera
                if (cameraMode == 1){
                    // Create ImageCapture
                    imageCapture = ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build()
                    camera = cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageCapture)
                }else{
                    imageAnalyzer = ImageAnalysis.Builder()
                        .build()
                        .also {
                            it.setAnalyzer(cameraExecutor, ClassImageAnalyzer { image ->
                               processImage(image)
                                var timeEnd = (System.nanoTime() - startStreaming)/0.000001
                                time_running_info.text = "$timeEnd ms"
                            })
                        }
                    camera = cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageAnalyzer)
                }
                preview?.setSurfaceProvider(viewFinder.createSurfaceProvider(camera?.cameraInfo))
            } catch(exc: Exception) {
                Timber.e("Use case binding failed : $exc")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @ExperimentalTime
    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return
        var timeEnd = (System.nanoTime() - startStreaming)
        time_running_info.text = "${timeEnd.milliseconds} ms"
        val photoFile = File(outputDirectory, "smart.io.picture.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(
            outputOptions, ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Timber.e("Photo capture failed: ${exc.message}")
                }
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val savedUri = Uri.fromFile(photoFile)
                    val imageBitmap:Bitmap = BitmapFactory.decodeFile(savedUri.path)

                    cameraPreview.setImageBitmap(imageBitmap)
                    processImage(imageBitmap)
                    camera_capture_button.visibility = View.GONE
                    stats_container.visibility = View.VISIBLE
                }
            })
    }
    private fun allPermissionsGranted() = false


    private fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let {
            File(it, resources.getString(R.string.app_name)).apply { mkdirs() } }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }

    companion object {
        private const val TAG = "CameraXBasic"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private var numThread = 6
        private const val IMMERSIVE_FLAG_TIMEOUT = 500L

    }
    private class ClassImageAnalyzer(private val listener: DetectionListener) : ImageAnalysis.Analyzer {
        fun Image.toBitmap(): Bitmap {
            //val planes = image.planes
            val yBuffer = planes[0].buffer // Y
            val uBuffer = planes[1].buffer // U
            val vBuffer = planes[2].buffer // V

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            //U and V are swapped
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
            val imageBytes = out.toByteArray()
            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        }
        private fun ByteBuffer.toByteArray(): ByteArray {
            rewind()
            val data = ByteArray(remaining())
            get(data)
            return data
        }
        @SuppressLint("UnsafeExperimentalUsageError")
        override fun analyze(imageProxy: ImageProxy) {
            val image: Image = imageProxy.image!!
            val buffer = imageProxy.planes[0].buffer
            listener(image.toBitmap())
            imageProxy.close()
        }
    }

    private  fun setUp(){
        threadsTextView = findViewById(R.id.frame_info)
        trackingOverlay = findViewById(R.id.tracking_overlay)
        tracker = SingleBoxTracker(applicationContext)
        detector = TFLiteObjectDetectionAPIModel.create(
            assets,
            modelFile,
            labelFile,
            apiInputSize,
            isQuantized
        )
        trackingOverlay!!.addCallback(
            object : OverlayView.DrawCallback {
                override fun drawCallback(canvas: Canvas) {
                    tracker!!.draw(canvas)
                }
            })
        startStreaming = System.nanoTime()
    }
    private fun processImage(image: Bitmap){
        trackingOverlay!!.postInvalidate()
        detector!!.setNumThreads(numThread)
        ++timestamp
        if (cameraMode == 1){
            cameraPreview.setImageBitmap(image)
            viewFinder.visibility = View.GONE
        }
        val currTimestamp = timestamp
        val cropSize: Int = apiInputSize
        val previewWidth: Int = image.width
        val previewHeight: Int = image.height
        val sensorOrientation = 0
        val croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        tracker!!.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation)
        frameToCropTransform = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, false
        )
        cropToFrameTransform = Matrix()
        frameToCropTransform!!.invert(cropToFrameTransform)
        val canvas = Canvas(croppedBitmap!!)
        if (image != null) {
            canvas.drawBitmap(image, frameToCropTransform!!, null)
        }
        tracker!!.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation)
        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap)
        if (saveShot) {
            val savedFile = File(
                outputDirectory,
                "smart.io.cropped.jpg"
            )
            val out = FileOutputStream(savedFile)
            croppedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
        }
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2.0f
        val startTime = SystemClock.uptimeMillis()
        val results = detector!!.recognizeImage(croppedBitmap)
        var minimumConfidence: Float = confidenceTf
        when (modeApi) {
            DetectorMode.TF_OD_API -> minimumConfidence =
                confidenceTf
        }
        val mappedRecognitions: MutableList<Recognition> = LinkedList()
        var count = 0

        for (result in results!!) {
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
            val location: RectF = result.location
            total_persons_info.text = "$totalCount"
            frame_info.text = count.toString()
            inference_info.text = "$lastProcessingTimeMs fps"
            if (location != null && result.confidence!! >= minimumConfidence) {
                if (result.title in classes) {
                    ++totalCount
                    ++count
                    category_layout.visibility = View.VISIBLE
                    category_info.setTextColor(resources.getColor(R.color.colorButton))
                    category.setTextColor(resources.getColor(R.color.colorButton))
                    category_info.text = "${result.title.toUpperCase()}"
                }else{
                    category_layout.visibility = View.VISIBLE
                    category_info.text = "${result.title.toUpperCase()}"
                }
                total_persons_info.text = "$totalCount"
                frame_info.text = count.toString()
                inference_info.text = "$lastProcessingTimeMs fps"
                canvas.drawRect(location, paint)
                cropToFrameTransform!!.mapRect(location)
                result.location = location
                mappedRecognitions.add(result)

            }
        }
        tracker!!.trackResults(mappedRecognitions, currTimestamp)
        trackingOverlay!!.postInvalidate()
    }
}

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {
    private val callbacks: MutableList<DrawCallback> =
        LinkedList()
    fun addCallback(callback: DrawCallback) {
        callbacks.add(callback)
    }
    @Synchronized
    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        for (callback in callbacks) {
            callback.drawCallback(canvas)
        }
    }
    interface DrawCallback {
        fun drawCallback(canvas: Canvas)
    }
}
