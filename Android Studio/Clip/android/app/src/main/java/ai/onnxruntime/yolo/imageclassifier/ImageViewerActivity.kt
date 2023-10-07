package ai.onnxruntime.yolo.imageclassifier

import ai.onnxruntime.yolo.imageclassifier.databinding.ActivityImageViewerBinding
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView

private const val TAG = "ImageViewerActivity"

class ImageViewerActivity : AppCompatActivity() {

    private lateinit var binding: ActivityImageViewerBinding
    lateinit var adapter: ImageAdapter
    private var numImages = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_image_viewer)

        val recyclerView = findViewById<RecyclerView>(R.id.recyclerView)
        recyclerView.setHasFixedSize(true)

        val layoutManager = GridLayoutManager(applicationContext, 2)
        recyclerView.layoutManager = layoutManager

        numImages = intent.getIntExtra("numImages", 10)
        val objectImage = intent.getByteArrayExtra("objectImage")!!

        adapter = ImageAdapter(arrayListOf())
        recyclerView.adapter = adapter

        binding = ActivityImageViewerBinding.inflate(layoutInflater)

        DatabaseConnection(objectImage, numImages, ::setImages).start()
    }

    private fun setImages(images: Array<ByteArray>) {
        runOnUiThread {
            images.forEach {
                adapter.addImage(BitmapFactory.decodeByteArray(it, 0, it.size))
            }
        }
    }
}