package ai.onnxruntime.yolo.imageclassifier

import ai.onnxruntime.*
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.*

private const val TAG = "ORTAnalyzer"

data class Result(
    var detectedBoxes: MutableList<FloatArray> = mutableListOf(),
    var detectedScore: MutableList<Float> = mutableListOf(),
    var detectedClasses: MutableList<Int> = mutableListOf(),
    var processTimeMs: Long = 0,
    var rawImage: Bitmap
)

data class Detection(
    val bbox: FloatArray, val objectClass: Int, val conf: Float
)

internal class ORTAnalyzer(
    private val viewWidth: Int,
    private val viewHeight: Int,
    private val ortSession: OrtSession?,
    private val callBack: (Result) -> Unit,
    private val writeImage: (Bitmap) -> Unit
) : ImageAnalysis.Analyzer {

    private fun postProcess(
        result: Array<Array<FloatArray>>, confThreshold: Float = 0.25f, iouThreshold: Float = 0.45f
    ): List<Detection> {
        // result is 1 x 84 x 8400
        // resized result is 8400 x 84
        val resizedResult = Array(8400) { detectionIndex ->
            FloatArray(84) { classIndex ->
                result[0][classIndex][detectionIndex]
            }
        }

        val candidates = mutableListOf<Detection>()
        resizedResult.forEach { candidate ->
            val maxConfClass =
                candidate.slice(4 until 84).indices.maxBy { classIndex -> candidate[classIndex + 4] }
            val classConf = candidate[maxConfClass + 4]
            if (classConf >= confThreshold) {
                candidates.add(
                    Detection(
                        candidate.slice(0 until 4).toFloatArray(), maxConfClass, classConf
                    )
                )
            }
        }
        Log.i(TAG, "Number of candidates is ${candidates.size}")

        // change bounding box coordinates to x1y1x2y2
        candidates.forEach { cxcywh2xyxy(it) }

        val detections = nonMaxSuppression(candidates, iouThreshold)
//        adjustToScreen(detections)
        return detections
    }

    private fun nonMaxSuppression(
        candidates: List<Detection>, iouThreshold: Float
    ): List<Detection> {
        val numBoxes = candidates.size
        val indicesSortedByScore = candidates.indices.sortedByDescending { candidates[it].conf }
        val considered = BooleanArray(numBoxes) { false }
        val filteredCandidates = mutableListOf<Detection>()

        for (i in 0 until numBoxes) {
            val currentAcceptingIndex = indicesSortedByScore[i]
            if (!considered[currentAcceptingIndex]) {
                considered[currentAcceptingIndex] = true
                filteredCandidates.add(candidates[currentAcceptingIndex])

                for (j in (i + 1) until numBoxes) {
                    val currentIndex = indicesSortedByScore[j]
                    if (calculateIOU(
                            candidates[currentAcceptingIndex].bbox, candidates[currentIndex].bbox
                        ) > iouThreshold
                    ) {
                        considered[currentIndex] = true
                    }
                }
            }
        }
        return filteredCandidates
    }

    /**
     * Converts coordinates in the form of xCenter, yCenter, width, height into edge coordinates
     * x1, y1, x2, y2
     * works in-place
     **/
    private fun cxcywh2xyxy(xywh: Detection) {
        val xCenter = xywh.bbox[0]
        val yCenter = xywh.bbox[1]
        val width = xywh.bbox[2]
        val height = xywh.bbox[3]
        xywh.bbox[0] = xCenter - width / 2
        xywh.bbox[1] = yCenter - height / 2
        xywh.bbox[2] = xCenter + width / 2
        xywh.bbox[3] = yCenter + height / 2
    }

    // Calculates the intersection over union of two boxes
    private fun calculateIOU(box1: FloatArray, box2: FloatArray): Float {
        val intersectionWidth = maxOf(minOf(box1[2], box2[2]) - maxOf(box1[0], box2[0]), 0f)
        val intersectionHeight = maxOf(minOf(box1[3], box2[3]) - maxOf(box1[1], box2[1]), 0f)

        val intersectionArea = intersectionHeight * intersectionWidth

        val box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        val box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        val unionArea = box1Area + box2Area - intersectionArea

        return intersectionArea / unionArea
    }

    private fun adjustToOriginalSize(detections: List<Detection>, height: Int, width: Int) {
        val xFactor = width.toFloat() / 640f
        val yFactor = height.toFloat() / 640f

        detections.forEach { detection ->
            detection.bbox[0] *= xFactor
            detection.bbox[2] *= xFactor
            detection.bbox[1] *= yFactor
            detection.bbox[3] *= yFactor
        }
    }

    private fun adjustToScreen(detections: List<Detection>) {
        Log.i(TAG, "width/height is ${viewWidth}, $viewHeight")
        val aspectRatio = viewHeight.toFloat() / viewWidth.toFloat()
        Log.i(TAG, "Aspect ratio is $aspectRatio")

        val factor: Float
        // Why not divided by 2?
        val sideMargin = (3f / 4f - 9f / 16f) / 2f

        if (aspectRatio >= 16f / 9f) {
            factor = viewWidth * 16f / 9f / 640f

            detections.forEach { detection ->
                detection.bbox[0] = detection.bbox[0] * factor * 0.75f - (sideMargin * viewWidth)
                detection.bbox[2] = detection.bbox[2] * factor * 0.75f - (sideMargin * viewWidth)
                detection.bbox[1] *= factor
                detection.bbox[3] *= factor
            }
            Log.i(TAG, "Width restricted")
        } else {
            factor = viewHeight / 640f

            detections.forEach { detection ->
                detection.bbox[0] =
                    detection.bbox[0] * factor * 0.75f - (sideMargin * viewHeight * 9f / 16f)
                detection.bbox[2] =
                    detection.bbox[2] * factor * 0.75f - (sideMargin * viewHeight * 9f / 16f)
                detection.bbox[1] *= factor
                detection.bbox[3] *= factor
            }
            Log.i(TAG, "Height restricted")
        }


    }

    // Rotate the image of the input bitmap
    private fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

//    private fun compressBitmap(bitmap: Bitmap): ByteArray {
//        val os = ByteArrayOutputStream()
//        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, os)
//        return os.toByteArray()
//    }

    override fun analyze(image: ImageProxy) {
        var timestamp = SystemClock.uptimeMillis()
        val imgBitmap = image.toBitmap().rotate(image.imageInfo.rotationDegrees.toFloat())
        val bitmap = Bitmap.createScaledBitmap(imgBitmap, IMAGE_SIZE_X, IMAGE_SIZE_Y, false)

        Log.i(TAG, "Scaled bitmap in ${SystemClock.uptimeMillis() - timestamp} ms")
        timestamp = SystemClock.uptimeMillis()

        val result = Result(rawImage=imgBitmap)

        val imgData = preProcess(bitmap)
        val inputName = ortSession?.inputNames?.iterator()?.next()
        val shape = longArrayOf(1, 3, 640, 640)
        val env = OrtEnvironment.getEnvironment()
        Log.i(TAG, "Preprocess done in ${SystemClock.uptimeMillis() - timestamp} ms")
        timestamp = SystemClock.uptimeMillis()
        env.use {
            val tensor = OnnxTensor.createTensor(env, imgData, shape)
            Log.i(TAG, "Tensor created in ${SystemClock.uptimeMillis() - timestamp} ms")
            timestamp = SystemClock.uptimeMillis()
            val startTime = SystemClock.uptimeMillis()
            tensor.use {
                val output = ortSession?.run(Collections.singletonMap(inputName, tensor))
                Log.i(TAG, "Inference done in ${SystemClock.uptimeMillis() - timestamp} ms")
                timestamp = SystemClock.uptimeMillis()
                output.use {
                    result.processTimeMs = SystemClock.uptimeMillis() - startTime
                    val rawOutput = output?.get(0)?.value

                    @Suppress("UNCHECKED_CAST") val detections =
                        postProcess(rawOutput as Array<Array<FloatArray>>)
                    adjustToOriginalSize(detections, imgBitmap.height, imgBitmap.width)


                    detections.forEach {
                        result.detectedBoxes.add(it.bbox)
                        result.detectedClasses.add(it.objectClass)
                        result.detectedScore.add(it.conf)
                    }
                    result.processTimeMs = timestamp - SystemClock.uptimeMillis()
                }
                Log.i(
                    TAG, "Postprocessing done in ${SystemClock.uptimeMillis() - timestamp} ms"
                )
                timestamp = SystemClock.uptimeMillis()
            }
        }
        callBack(result)

        image.close()
    }

    protected fun finalize() {
        ortSession?.close()
    }
}