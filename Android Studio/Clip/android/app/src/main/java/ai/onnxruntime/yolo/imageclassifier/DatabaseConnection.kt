package ai.onnxruntime.yolo.imageclassifier

import android.util.Log
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import java.lang.Integer.min
import java.net.InetSocketAddress
import java.net.Socket

private const val TAG = "DatabaseConnection"
private const val PORT = 8990

private const val IP = "10.34.58.149"

class DatabaseConnection(
    private val objectImage: ByteArray,
    private val numImages: Int,
    private val callback: (Array<ByteArray>) -> (Unit)
) : Thread() {

    private lateinit var dataInputStream: DataInputStream
    private lateinit var dataOutputStream: DataOutputStream
    private lateinit var socket: Socket

    override fun run() {
        try {
            socket = Socket()
            Log.i(TAG, "Trying to connect")
            socket.connect(InetSocketAddress(IP, PORT), 1000)
            val inputStream = socket.getInputStream()
            dataInputStream = DataInputStream(inputStream)
            val outputStream = socket.getOutputStream()
            dataOutputStream = DataOutputStream(outputStream)
            Log.i(TAG, "Connection established")
        } catch (ex: IOException) {
            Log.e(TAG, ex.toString())
            return
        }

        dataOutputStream.writeInt(objectImage.size)
        dataOutputStream.write(objectImage)
        dataOutputStream.writeInt(numImages)
        dataOutputStream.flush()

        Log.i(TAG, "Classes and num images written")

        val images = Array(numImages) { byteArrayOf(0) }

        for (i in 0 until numImages) {
            val msgLength = dataInputStream.readInt()
            Log.i(TAG, "msg length is $msgLength")
            images[i] = ByteArray(msgLength)
            Log.i(TAG, "Reading data now")
            var bytesReceived = 0
            while (bytesReceived < msgLength) {
                val remaining = msgLength - bytesReceived
                val bytesToRead = min(4096, remaining)
                val newBytesReceived = dataInputStream.read(images[i], bytesReceived, bytesToRead)

                // Safeguard to keep bytesReceived from decrementing and breaking everything
                if (newBytesReceived < 0) {
                    break
                }

                bytesReceived += newBytesReceived
            }
            Log.i(TAG, "Received $bytesReceived bytes, deficit of ${msgLength - bytesReceived}")
        }
        callback(images)
    }
}