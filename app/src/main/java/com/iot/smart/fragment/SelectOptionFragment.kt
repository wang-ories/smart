package com.iot.smart.fragment

import android.content.Intent
import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import com.iot.smart.DetectionActivity
import com.iot.smart.R


private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

class SelectOptionFragment : Fragment() {
    private var param1: String? = null
    private var param2: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)
            param2 = it.getString(ARG_PARAM2)
        }
    }
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        var rootView =  inflater.inflate(R.layout.fragment_select_option, container, false)
        var cameraOption = rootView.findViewById<Button>(R.id.camera_option)
        cameraOption.setOnClickListener {
            var i = Intent(rootView.context, DetectionActivity::class.java)
            i.putExtra("cameraMode", 0)
            startActivity(i)
        }
        var videoOption = rootView.findViewById<Button>(R.id.video_option)
        videoOption.setOnClickListener {
            var i = Intent(rootView.context, DetectionActivity::class.java)
            i.putExtra("cameraMode", 1)
            startActivity(i)
        }
       /* var croppedBitmap = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(croppedBitmap)
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2.0f
        canvas.drawRect(RectF(100F, 100F, 200F, 200F), paint)

        */

        return rootView
    }

    companion object {
        @JvmStatic
        fun newInstance(param1: String, param2: String) =
            SelectOptionFragment().apply {
                arguments = Bundle().apply {
                    putString(ARG_PARAM1, param1)
                    putString(ARG_PARAM2, param2)
                }
            }
    }
}
