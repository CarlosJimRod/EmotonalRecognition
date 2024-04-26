/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.kotlin.facedetector

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.demo.BitmapUtils
import com.google.mlkit.vision.demo.GraphicOverlay
import com.google.mlkit.vision.demo.kotlin.VisionProcessorBase
import com.google.mlkit.vision.demo.tflite.Classifier
import com.google.mlkit.vision.demo.tflite.EmotionClassifier
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import java.util.Locale

/** Face Detector Demo.  */
class FaceDetectorProcessor(
    context: Context,
    detectorOptions: FaceDetectorOptions?,
    val callback: Callback
) :
    VisionProcessorBase<List<Face>>(context) {

    private val detector: FaceDetector

    private val context = context

    private var classifier: Classifier

    init {
        val options = detectorOptions
            ?: FaceDetectorOptions.Builder()
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                .enableTracking()
                .build()

        detector = FaceDetection.getClient(options)
        classifier = EmotionClassifier(context as Activity)
    }

    override fun stop() {
        super.stop()
        detector.close()
    }

    override fun detectInImage(image: InputImage): Task<List<Face>> {
        return detector.process(image)
    }

    @RequiresApi(Build.VERSION_CODES.LOLLIPOP)
    override fun onSuccess(
        results: List<Face>,
        image: Bitmap?,
        graphicOverlay: GraphicOverlay
    ) {
        for (face in results) {
            graphicOverlay.add(FaceGraphic(graphicOverlay, face, context))
            image?.let { bitmap ->
                val croppedImage: Bitmap = BitmapUtils.cropBitmap(bitmap, face.boundingBox)
                val faceExpressions: List<Classifier.Recognition> =
                    classifier.recognizeImage(croppedImage)
                if (faceExpressions.isNotEmpty()) {
                    callback.onSuccess(faceExpressions.first())
                    Log.d("_dev_", faceExpressions.first().title)
                }
            }
        }
    }

    override fun onFailure(e: Exception) {
        callback.onError(e)
    }

    interface Callback {
        fun onSuccess(faceExpression: Classifier.Recognition)
        fun onError(e: Exception)
    }
}
