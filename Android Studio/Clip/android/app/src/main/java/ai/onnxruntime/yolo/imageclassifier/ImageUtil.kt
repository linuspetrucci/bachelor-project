package ai.onnxruntime.yolo.imageclassifier

import android.graphics.Bitmap
import java.nio.FloatBuffer

const val DIM_BATCH_SIZE = 1
const val DIM_PIXEL_SIZE = 3
const val IMAGE_SIZE_X = 640
const val IMAGE_SIZE_Y = 640

fun preProcess(bitmap: Bitmap): FloatBuffer {
    val imgData = FloatBuffer.allocate(
        DIM_BATCH_SIZE * DIM_PIXEL_SIZE * IMAGE_SIZE_X * IMAGE_SIZE_Y
    )
    imgData.rewind()
    val stride = IMAGE_SIZE_X * IMAGE_SIZE_Y
    val bmpData = IntArray(stride)
    bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    for (i in 0 until IMAGE_SIZE_X) {
        for (j in 0 until IMAGE_SIZE_Y) {
            val idx = IMAGE_SIZE_Y * i + j
            val pixelValue = bmpData[idx]
            imgData.put(idx, (pixelValue shr 16 and 0xFF) / 255f)
            imgData.put(idx + stride, (pixelValue shr 8 and 0xFF) / 255f)
            imgData.put(idx + stride * 2, (pixelValue and 0xFF) / 255f)
        }
    }

    imgData.rewind()
    return imgData
}
