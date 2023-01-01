package com.example.speed_sheet3

import android.Manifest.permission.READ_EXTERNAL_STORAGE
import android.content.ContentUris
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager.PERMISSION_DENIED
import android.database.Cursor
import android.graphics.Bitmap
import android.graphics.Bitmap.CompressFormat.JPEG
import android.graphics.Bitmap.Config.RGB_565
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.StrictMode
import android.provider.DocumentsContract
import android.provider.MediaStore
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat.requestPermissions
import androidx.core.content.ContextCompat
import okhttp3.*
import okhttp3.RequestBody.create
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit


class MainActivity : AppCompatActivity() {
    var selectedImagePath: String? = null
    var fileName = "false.jpg"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if (ContextCompat.checkSelfPermission(this@MainActivity, READ_EXTERNAL_STORAGE) == PERMISSION_DENIED) {
            requestPermissions(this@MainActivity, arrayOf(READ_EXTERNAL_STORAGE), 101)
        } else {
            Toast.makeText(this@MainActivity, "Permission already granted", Toast.LENGTH_SHORT)
                .show()
        }
    }

    fun selectImage(v: View?) {
        val intent = Intent()
        intent.type = "*/*"
        intent.action = Intent.ACTION_GET_CONTENT
        startActivityForResult(intent, 0)
    }

    override fun onActivityResult(reqCode: Int, resCode: Int, data: Intent?) {
        super.onActivityResult(reqCode, resCode, data)
        if (resCode == RESULT_OK && data != null) {
            val uri = data.data
            selectedImagePath = getPath(applicationContext, uri)
            val imgPath = findViewById<EditText>(R.id.imgPath)
            imgPath.setText(selectedImagePath)
            Toast.makeText(applicationContext, selectedImagePath, Toast.LENGTH_LONG).show()

            val stream = ByteArrayOutputStream()
            val options = BitmapFactory.Options()
            options.inPreferredConfig = RGB_565
            val bitmap = BitmapFactory.decodeFile(selectedImagePath, options)
            bitmap.compress(JPEG, 100, stream)
            val imageView = findViewById<ImageView>(R.id.imageView)
            imageView.setImageBitmap(bitmap)
        }
    }

    fun itemClicked(v: View) {
        val checkBox = v as CheckBox
        fileName = if (!checkBox.isChecked) {
            "false.jpg"
        } else {
            "true.jpg"
        }
    }

    fun connectToServer(v: View?) {
        if (selectedImagePath != null) {
            val postUrl = "http://10.0.2.2:8000/"
            val stream = ByteArrayOutputStream()
            val options = BitmapFactory.Options()
            options.inPreferredConfig = RGB_565
            println("PATH" + selectedImagePath)
            val bitmap = BitmapFactory.decodeFile(selectedImagePath, options)
            //val bitmap = BitmapFactory.decodeStream(ByteArrayInputStream(stream.toByteArray()))
            println("BITMAP " + bitmap.height)

            bitmap.compress(JPEG, 100, stream)

            val byteArray = stream.toByteArray()
            val postBodyImage: RequestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "image",
                    fileName,
                    create(MediaType.parse("image/*jpg"), byteArray)
                )
                .build()
            val responseText = findViewById<TextView>(R.id.responseText)
            responseText.text = "Please wait ..."
            makePostRequest(postUrl, postBodyImage)
        }
    }

    private fun makePostRequest(postUrl: String?, postBody: RequestBody?) {
        StrictMode.setThreadPolicy(StrictMode.ThreadPolicy.Builder().permitAll().build())
        val client = OkHttpClient().newBuilder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .writeTimeout(100, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .build()
        val request = Request.Builder()
            .url(postUrl)
            .post(postBody)
            .build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                call.cancel()
                runOnUiThread {
                    val responseText = findViewById<TextView>(R.id.responseText)
                    responseText.text = "Failed to Connect to Server"
                }
            }

            @Throws(IOException::class)
            override fun onResponse(call: Call, response: Response) {
                runOnUiThread {
                    val responseText = findViewById<TextView>(R.id.responseText)
                    try {
                        responseText.text = response.body().string()
                    } catch (e: IOException) {
                        e.printStackTrace()
                    }
                }
            }
        })
    }

    companion object {

        fun getPath(context: Context, uri: Uri?): String? {
            if (DocumentsContract.isDocumentUri(context, uri)) {
                if (isExternalStorageDocument(uri)) {
                    val docId = DocumentsContract.getDocumentId(uri)
                    val split = docId.split(":").toTypedArray()
                    val type = split[0]
                    if ("primary".equals(type, ignoreCase = true)) {
                        return Environment.getExternalStorageDirectory().toString() + "/" + split[1]
                    }
                } else if (isDownloadsDocument(uri)) {
                    val id = DocumentsContract.getDocumentId(uri)
                    println(id)
                    val contentUri = ContentUris.withAppendedId(
                        Uri.parse("content://downloads/public_downloads"),
                        java.lang.Long.valueOf(id)
                    )
                    return getData(context, contentUri, null, null)
                } else if (isMediaDocument(uri)) {
                    val docId = DocumentsContract.getDocumentId(uri)
                    val split = docId.split(":").toTypedArray()
                    val type = split[0]
                    var contentUri: Uri? = null
                    if ("image" != type) {
                        if ("video" == type) {
                            contentUri = MediaStore.Video.Media.EXTERNAL_CONTENT_URI
                        } else if ("audio" == type) {
                            contentUri = MediaStore.Audio.Media.EXTERNAL_CONTENT_URI
                        }
                    } else {
                        contentUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                    }
                    val selection = "_id=?"
                    val selectionArgs = arrayOf(
                        split[1]
                    )
                    return getData(context, contentUri, selection, selectionArgs)
                }
            } else if ("content".equals(uri!!.scheme, ignoreCase = true)) {
                return getData(context, uri, null, null)
            } else if ("file".equals(uri.scheme, ignoreCase = true)) {
                return uri.path
            }
            return null
        }

        private fun getData(
            context: Context, uri: Uri?, selection: String?,
            selectionArgs: Array<String>?
        ): String? {
            var cursor: Cursor? = null
            val column = "_data"
            val projection = arrayOf(
                column
            )
            try {
                cursor = context.contentResolver.query(
                    uri!!, projection, selection, selectionArgs,
                    null
                )
                if (cursor != null && cursor.moveToFirst()) {
                    val columnIndex = cursor.getColumnIndexOrThrow(column)
                    return cursor.getString(columnIndex)
                }
            } finally {
                cursor?.close()
            }
            return null
        }

        private fun isExternalStorageDocument(uri: Uri?): Boolean {
            return "com.android.externalstorage.documents" == uri!!.authority
        }

        private fun isDownloadsDocument(uri: Uri?): Boolean {
            return "com.android.providers.downloads.documents" == uri!!.authority
        }

        private fun isMediaDocument(uri: Uri?): Boolean {
            return "com.android.providers.media.documents" == uri!!.authority
        }
    }
}