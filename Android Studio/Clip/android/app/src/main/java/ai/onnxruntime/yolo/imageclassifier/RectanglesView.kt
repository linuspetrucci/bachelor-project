package ai.onnxruntime.yolo.imageclassifier

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.JsonReader
import android.util.Log
import android.view.MotionEvent
import android.view.View
import com.google.android.material.snackbar.Snackbar
import java.io.InputStreamReader
import java.lang.Float.max
import java.lang.Float.min
import java.text.DecimalFormat

private const val TAG = "RectanglesView"

class RectanglesView(
    context: Context,
    attrs: AttributeSet,
) : View(context, attrs) {

    private val paint: Paint = Paint()
    private val rawRectangles: MutableList<FloatArray> = mutableListOf()
    private val rectangles: MutableList<FloatArray> = mutableListOf()
    private val detectionClasses: MutableList<Int> = mutableListOf()
    private val detectionConfidences: MutableList<Float> = mutableListOf()
    private lateinit var callBack: (FloatArray) -> Unit
    private val classNames = HashMap<Int, String>()
    private val decimalFormat = DecimalFormat("##")
    private var colors: IntArray = intArrayOf(
        Color.parseColor("#3d96ba"),
        Color.parseColor("#38c290"),
        Color.parseColor("#52c421"),
        Color.parseColor("#E7AC17"),
        Color.parseColor("#E73e17")
    )

    init {
        paint.color = colors[0]
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 6F
        paint.textSize = 60F

        val jsonReader =
            JsonReader(InputStreamReader(resources.openRawResource(R.raw.image_classes)))
        jsonReader.use {
            jsonReader.beginObject()
            while (jsonReader.hasNext()) {
                classNames[jsonReader.nextName().toInt()] = jsonReader.nextString()
            }
            jsonReader.endObject()
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        canvas.drawColor(Color.TRANSPARENT)
        paint.style = Paint.Style.STROKE
        rectangles.forEachIndexed { index, rectangle ->
            paint.color = colors[index % colors.size]
            canvas.drawRoundRect(rectangle[0], rectangle[1], rectangle[2], rectangle[3], 10f, 10f, paint)
        }

        paint.style = Paint.Style.FILL
        detectionClasses.forEachIndexed { index, detectedClass ->
            paint.color = colors[index % colors.size]
            classNames[detectedClass]?.let {
                canvas.drawText(
                    it.plus(" ").plus(
                        decimalFormat.format(
                            detectionConfidences[index] * 100F
                        ).plus("%")
                    ), rectangles[index][0] + 5F, rectangles[index][1] + 60F, paint
                )
            }
        }
        postInvalidate()
    }


    fun setDetections(result: Result) {
        val adjustedBoxes =
            adjustToScreen(result.detectedBoxes, result.rawImage.height, result.rawImage.width)

        rawRectangles.clear()
        rawRectangles.addAll(result.detectedBoxes)

        rectangles.clear()
        rectangles.addAll(adjustedBoxes)

        detectionClasses.clear()
        detectionClasses.addAll(result.detectedClasses)

        detectionConfidences.clear()
        detectionConfidences.addAll(result.detectedScore)
    }

    private fun adjustToScreen(
        boxes: MutableList<FloatArray>,
        height: Int,
        width: Int
    ): MutableList<FloatArray> {

        val adjustedBoxes = mutableListOf<FloatArray>()

        val screenWidth = this.width.toFloat()
        val screenHeight = this.height.toFloat()

        val aspectRatioScreen = screenHeight / screenWidth
        val aspectRatioImage = height.toFloat() / width.toFloat()

        val sideMargin = (1 / aspectRatioImage - 9f / 16f)

        if (aspectRatioScreen >= 16f / 9f) {
            val factor = screenWidth * 16f / 9f / 640f

            boxes.forEach { box ->
                val x1 = max(box[0] * factor - (sideMargin * screenWidth), 0f)
                val x2 = min(box[2] * factor - (sideMargin * screenWidth), screenWidth)
                val y1 = max(box[1] * factor, 0f)
                val y2 = min(box[3] * factor, screenHeight)
                adjustedBoxes.add(floatArrayOf(x1, y1, x2, y2))
            }
            Log.i(TAG, "Width restricted")
        } else {
            val factor = screenHeight / 640f

            boxes.forEach { box ->
                val x1 = max(box[0] * factor - (sideMargin * screenHeight * 9f / 16f), 0f)
                val x2 = min(box[2] * factor - (sideMargin * screenHeight * 9f / 16f), screenWidth)
                val y1 = max(box[1] * factor, 0f)
                val y2 = min(box[3] * factor, screenHeight)
                adjustedBoxes.add(floatArrayOf(x1, y1, x2, y2))
            }
            Log.i(TAG, "Height restricted")
        }
        return adjustedBoxes
    }

    fun setCallBack(callBack: (FloatArray) -> Unit) {
        this.callBack = callBack
    }

    override fun performClick(): Boolean {
        return super.performClick()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        super.onTouchEvent(event)
        if (event.action != MotionEvent.ACTION_DOWN) {
            return true
        }
        performClick()
        Log.i(TAG, "Touch coordinates are ${event.x}, ${event.y}")
        val clickedRectangle = getCoordinatesOfClickedRectangle(event.x, event.y)
        Log.i(TAG, "clicked class is $clickedRectangle")
        if (clickedRectangle == null) {
            Snackbar.make(this, "No bounding box was selected", Snackbar.LENGTH_SHORT).show()
            return true
        }
        callBack(clickedRectangle)
        return true
    }

    private fun getCoordinatesOfClickedRectangle(x: Float, y: Float): FloatArray? {
        if (rectangles.isEmpty()) {
            return null
        }
        rectangles.forEachIndexed { index, it ->
            if (it[2] >= x && x >= it[0] && it[3] >= y && y >= it[1]) {
                return rawRectangles[index]
            }
        }
        return null
    }
}