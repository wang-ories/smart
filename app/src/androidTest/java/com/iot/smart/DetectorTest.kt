/*
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.iot.smart

import android.graphics.*
import android.util.Size
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iot.smart.tf.Classifier
import com.iot.smart.tf.TFLiteObjectDetectionAPIModel
import com.iot.smart.utils.ImageUtils
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.IOException
import java.util.*

@RunWith(AndroidJUnit4::class)
class DetectorTest {
    private var detector: Classifier? = null
    private var croppedBitmap: Bitmap? = null
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private  val inputSize = 300
    private val isQuantized = true
    private  val modelFile = "detect.tflite"
    private  val labelFile = "file:///android_asset/labelmap.txt"
    private val imageSize = Size(640, 480)

    @Before
    @Throws(IOException::class)
    fun setUp() {
        val assetManager =
            InstrumentationRegistry.getInstrumentation().context
                .assets
        detector = TFLiteObjectDetectionAPIModel.create(
            assetManager,
            modelFile,
            labelFile,
            inputSize,
            isQuantized
        )
        val cropSize = inputSize
        val previewWidth = imageSize.width
        val previewHeight = imageSize.height
        val sensorOrientation = 0
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        frameToCropTransform = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, false
        )
        cropToFrameTransform = Matrix()
        frameToCropTransform!!.invert(cropToFrameTransform)
    }

    @Test
    @Throws(Exception::class)
    fun detectionResultsShouldNotChange() {
        val canvas = Canvas(croppedBitmap!!)
        canvas.drawBitmap(loadImage("table.jpg"), frameToCropTransform!!, null)
        val results: List<Classifier.Recognition> = detector!!.recognizeImage(croppedBitmap) as List<Classifier.Recognition>
        val expected: List<Classifier.Recognition> =
            loadRecognitions("table_results.txt")
        for (target in expected) {
            // Find a matching result in results
            var matched = false
            for (item in results) {
                val bbox = RectF()
                cropToFrameTransform!!.mapRect(bbox, item.getLocation())
                if (item.title.equals(target.title)
                    && matchBoundingBoxes(bbox, target.getLocation())
                    && matchConfidence(
                        item.confidence!!,
                        target.confidence!!
                    )
                ) {
                    matched = true
                    break
                }
            }
            assert(matched)
        }
    }

    companion object {
        // Confidence tolerance: absolute 1%
        private fun matchConfidence(a: Float, b: Float): Boolean {
            return Math.abs(a - b) < 0.01
        }
        // Bounding Box tolerance: overlapped area > 95% of each one
        private fun matchBoundingBoxes(a: RectF, b: RectF): Boolean {
            val areaA = a.width() * a.height()
            val areaB = b.width() * b.height()
            val overlapped = RectF(
                Math.max(a.left, b.left),
                Math.max(a.top, b.top),
                Math.min(a.right, b.right),
                Math.min(a.bottom, b.bottom)
            )
            val overlappedArea = overlapped.width() * overlapped.height()
            return overlappedArea > 0.95 * areaA && overlappedArea > 0.95 * areaB
        }

        @Throws(Exception::class)
        private fun loadImage(fileName: String): Bitmap {
            val assetManager =
                InstrumentationRegistry.getInstrumentation().context
                    .assets
            val inputStream = assetManager.open(fileName)
            return BitmapFactory.decodeStream(inputStream)
        }
        @Throws(Exception::class)
        private fun loadRecognitions(fileName: String): List<Classifier.Recognition> {
            val assetManager =
                InstrumentationRegistry.getInstrumentation().context
                    .assets
            val inputStream = assetManager.open(fileName)
            val scanner = Scanner(inputStream)
            val result: MutableList<Classifier.Recognition> =
                ArrayList<Classifier.Recognition>()
            while (scanner.hasNext()) {
                var category = scanner.next()
                category = category.replace('_', ' ')
                if (!scanner.hasNextFloat()) {
                    break
                }
                val left = scanner.nextFloat()
                val top = scanner.nextFloat()
                val right = scanner.nextFloat()
                val bottom = scanner.nextFloat()
                val boundingBox = RectF(left, top, right, bottom)
                val confidence = scanner.nextFloat()
                val recognition = Classifier.Recognition(null, category, confidence, boundingBox)
                result.add(recognition)
            }
            return result
        }
    }
}