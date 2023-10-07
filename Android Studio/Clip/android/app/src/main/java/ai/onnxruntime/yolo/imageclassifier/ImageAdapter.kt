package ai.onnxruntime.yolo.imageclassifier

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView

private const val TAG = "ImageAdapter"

class ImageAdapter(private val images: ArrayList<Bitmap>) :
    RecyclerView.Adapter<ImageAdapter.ViewHolder>() {
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val imageView: ImageView

        init {
            imageView = view.findViewById(R.id.imageView)
        }

    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.image_cell, parent, false)
        return ViewHolder(view)
    }

    override fun getItemCount(): Int {
        return images.size
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        Log.i(TAG, "Bound image $position")
        holder.imageView.adjustViewBounds = true
        holder.imageView.setImageBitmap(images[position])
        holder.imageView.setBackgroundColor(Color.RED)
    }

    fun addImage(newImage: Bitmap) {
        images.add(newImage)
        notifyItemInserted(itemCount - 1)
    }

}
