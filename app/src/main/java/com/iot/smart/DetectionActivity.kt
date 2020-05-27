package com.iot.smart

import android.Manifest
import android.app.Fragment
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.Image.Plane
import android.media.ImageReader
import android.os.*
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.iot.smart.tf.Classifier
import com.iot.smart.tf.TFLiteObjectDetectionAPIModel
import com.iot.smart.utils.ImageUtils
import timber.log.Timber

class DetectionActivity : AppCompatActivity(), ImageReader.OnImageAvailableListener{

    private val PERMISSION_CAMERA = Manifest.permission.CAMERA
    private val PERMISSIONS_REQUEST = 1

    // Configuration values for the prepackaged SSD model.
    private val TF_OD_API_INPUT_SIZE = 300
    private val TF_OD_API_IS_QUANTIZED = true
    private val TF_OD_API_MODEL_FILE = "detect.tflite"
    private val TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt"

    private var previewWidth = 0
    private var previewHeight = 0
    private val debug = false
    private val handler: Handler? = null
    private val handlerThread: HandlerThread? = null
    private var useCamera2API = false
    private var isProcessingFrame = false
    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    private var rgbBytes: IntArray? = null
    private var yRowStride = 0
    private var postInferenceCallback: Runnable? = null
    private var imageConverter: Runnable? = null
    private val MODEL_INPUT_SIZE = 300

    private val IMAGE_SIZE = Size(640, 480)
    private  var detector: Classifier? = null
    private var croppedBitmap: Bitmap? = null
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private var hasPermission  = true

    // Minimum detection confidence to track a detection.
    private val MINIMUM_CONFIDENCE_TF_OD_API = 0.5f
    private val MAINTAIN_ASPECT = false
    private val DESIRED_PREVIEW_SIZE = Size(640, 480)
    private val SAVE_PREVIEW_BITMAP = false
    private val TEXT_SIZE_DIP = 10f


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detection)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            hasPermission = checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED
        }
        if (!hasPermission){
            requestPermission()
        }
        setFragment()
            /* try {
            detector = TFLiteObjectDetectionAPIModel.create(
                applicationContext.assets,
                TF_OD_API_MODEL_FILE,
                TF_OD_API_LABELS_FILE,
                TF_OD_API_INPUT_SIZE,
                TF_OD_API_IS_QUANTIZED
            )


        } catch (e: IOException) {
            e.printStackTrace()
            Timber.e(
                e,
                "Exception initializing classifier!"
            )
            val toast = Toast.makeText(
                applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT
            )
            toast.show()
            finish()
        }
        val cropSize: Int = MODEL_INPUT_SIZE
        val previewWidth: Int = IMAGE_SIZE.width
        val previewHeight: Int = IMAGE_SIZE.height
        val sensorOrientation = 0
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)

        frameToCropTransform = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, false
        )
        cropToFrameTransform = Matrix()
        frameToCropTransform!!.invert(cropToFrameTransform)

        val canvas = Canvas()
        loadImage("face.jpeg")?.let {
            canvas.drawBitmap(
                it,
                frameToCropTransform!!,
                null
            )
        }
        val results: List<Classifier.Recognition?>? =
            detector!!.recognizeImage(croppedBitmap)
        Log.i("Huang", results.toString())

             */

    }
    private fun requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(
                    applicationContext,
                    "Camera permission is required for this demo",
                    Toast.LENGTH_LONG
                )
                    .show()
            }
            requestPermissions(
                arrayOf(PERMISSION_CAMERA),
                PERMISSIONS_REQUEST
            )
        }
    }
    @Throws(Exception::class)
    private fun loadImage(fileName: String): Bitmap? {
        val inputStream = assets.open(fileName)
        return BitmapFactory.decodeStream(inputStream)
    }
    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum class DetectorMode {
        TF_OD_API
    }

    /** Callback for Camera2 API  */
    override fun onImageAvailable(reader: ImageReader) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return
        }
        if (rgbBytes == null) {
            rgbBytes = IntArray(previewWidth * previewHeight)
        }
        try {
            val image = reader.acquireLatestImage() ?: return
            if (isProcessingFrame) {
                image.close()
                return
            }
            isProcessingFrame = true
            Trace.beginSection("imageAvailable")
            val planes = image.planes
            fillBytes(planes, yuvBytes)
            yRowStride = planes[0].rowStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride
            imageConverter = Runnable {
                yuvBytes[0]?.let {
                    ImageUtils.convertYUV420ToARGB8888(
                        it,
                        yuvBytes.get(1)!!,
                        yuvBytes.get(2)!!,
                        previewWidth,
                        previewHeight,
                        yRowStride,
                        uvRowStride,
                        uvPixelStride,
                        rgbBytes!!
                    )
                }
            }
            postInferenceCallback = Runnable {
                image.close()
                isProcessingFrame = false
            }
            processImage()
        } catch (e: java.lang.Exception) {
            Timber.e(e, "Exception!")
            Trace.endSection()
            return
        }
        Trace.endSection()
    }
    private fun fillBytes(
        planes: Array<Plane>,
        yuvBytes: Array<ByteArray?>
    ) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                Timber.d(
                    "Initializing buffer %d at size %d",
                    i,
                    buffer.capacity()
                )
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer[yuvBytes[i]]
        }
    }
    private fun setFragment() {
        val cameraId: String? = this!!.chooseCamera()
        /*val fragment: Fragment
        if (useCamera2API) {
            val camera2Fragment: CameraConnectionFragment = CameraConnectionFragment.newInstance(
                object : ConnectionCallback() {
                    fun onPreviewSizeChosen(
                        size: Size,
                        rotation: Int
                    ) {
                        previewHeight = size.height
                        previewWidth = size.width
                        onPreviewSizeChosen(size, rotation)
                    }
                },
                this,
                getLayoutId(),
                getDesiredPreviewFrameSize()
            )
            camera2Fragment.setCamera(cameraId)
            fragment = camera2Fragment
        } else {
            fragment =
                LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize())
        }
        fragmentManager.beginTransaction().replace(R.id.container, fragment).commit()

         */
    }
    private fun processImage() {
        Log.i("Huang", "process image".toString())

        // No mutex needed as this method is not reentrant.
       /* if (computingDetection) {
            readyForNextImage()
            return
        }
        computingDetection = true
        Timber.i("Preparing image  for detection in bg thread.")
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)
        readyForNextImage()
        val canvas = Canvas(croppedBitmap!!)
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform!!, null)
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap!!)
        }

        */
        runInBackground(
            Runnable {
                Timber.i("Running detection on image")
                detector = TFLiteObjectDetectionAPIModel.create(
                    applicationContext.assets,
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_INPUT_SIZE,
                    TF_OD_API_IS_QUANTIZED
                )
                val cropSize: Int = MODEL_INPUT_SIZE
                val previewWidth: Int = IMAGE_SIZE.width
                val previewHeight: Int = IMAGE_SIZE.height
                val sensorOrientation = 0
                croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)

                frameToCropTransform = ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropSize, cropSize,
                    sensorOrientation, false
                )
                cropToFrameTransform = Matrix()
                frameToCropTransform!!.invert(cropToFrameTransform)

                val canvas = Canvas()
                loadImage("face.jpeg")?.let {
                    canvas.drawBitmap(
                        it,
                        frameToCropTransform!!,
                        null
                    )
                }
                val results: List<Classifier.Recognition?>? =
                    detector!!.recognizeImage(croppedBitmap)
                Log.i("Huang", results.toString())


                /* val startTime = SystemClock.uptimeMillis()
            val results: List<Classifier.Recognition?>? =
                detector!!.recognizeImage(croppedBitmap)

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap!!)
            val canvas = Canvas(cropCopyBitmap)
            val paint = Paint()
            paint.color = Color.RED
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f
            var minimumConfidence: Float =
                MINIMUM_CONFIDENCE_TF_OD_API
            when (MODE) {
                DetectorMode.TF_OD_API -> minimumConfidence =
                    MINIMUM_CONFIDENCE_TF_OD_API
            }
            val mappedRecognitions: MutableList<Classifier.Recognition?> =
                LinkedList<Classifier.Recognition?>()
            for (result in results!!) {
                val location: RectF = result?.getLocation() ?:
                if (location != null && result.getConfidence() >= minimumConfidence) {
                    canvas.drawRect(location, paint)
                    cropToFrameTransform!!.mapRect(location)
                    result.setLocation(location)
                    mappedRecognitions.add(result)
                }
            }
            tracker.trackResults(mappedRecognitions, currTimestamp)
            trackingOverlay.postInvalidate()
            computingDetection = false

                 */
                runOnUiThread {
                    /*showFrameInfo(previewWidth.toString() + "x" + previewHeight)
                    showCropInfo(
                        cropCopyBitmap.getWidth()
                            .toString() + "x" + cropCopyBitmap.getHeight()
                    )
                    showInference(lastProcessingTimeMs.toString() + "ms")

                     */
                    Log.i("Huang", results.toString())

                }
            })
    }

    @Synchronized
    private fun runInBackground(r: Runnable?) {
        handler?.post(r)
    }

    private fun chooseCamera(): String? {
        val manager =  getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            for (cameraId in manager.cameraIdList) {
                val characteristics =  manager.getCameraCharacteristics(cameraId)
                // We don't use a front facing camera in this sample.
                val facing = characteristics.get(
                    CameraCharacteristics.LENS_FACING
                )
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue
                }
                val map = characteristics.get( CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                    ?: continue
                useCamera2API = (facing == CameraCharacteristics.LENS_FACING_EXTERNAL
                        || isHardwareLevelSupported(
                    characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL
                ))
                Timber.i(
                    "Camera API lv2?: %s",
                    useCamera2API
                )
                return cameraId
            }
        } catch (e: CameraAccessException) {
            Timber.e(
                e,
                "Not allowed to access camera"
            )
        }
        return null
    }
    // Returns true if the device supports the required hardware level, or better.
    private fun isHardwareLevelSupported(
        characteristics: CameraCharacteristics, requiredLevel: Int
    ): Boolean {
        val deviceLevel =
            characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)!!
        return if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            requiredLevel == deviceLevel
        } else requiredLevel <= deviceLevel
        // deviceLevel is not LEGACY, can use numerical sort
    }

}
