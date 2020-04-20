//#region IMPORTS
package org.pytorch.helloworld;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.io.FileWriter;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.res.TypedArrayUtils;

//#endregion

public class MainActivity extends AppCompatActivity {

  // for permission requests
  public static final int REQUEST_PERMISSION = 300;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    get_permissions();

    Bitmap bitmap = null;
    Bitmap bitmap_orig = null;
    Module module = null;
    int[] target_shape;
    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      bitmap = BitmapFactory.decodeStream(getAssets().open("ch08_20191007120955_25.JPG"));
      //bitmap = getResizedBitmap(bitmap_orig, 224, 224);

//     target_shape = getResizeTarget(bitmap_orig, 224, 224);
//     int target_height = target_shape[0];
//     int target_width = target_shape[1];
//     bitmap = getResizedBitmap(bitmap_orig,  target_width, target_height);
//     bitmap = cropBitMap(bitmap, 224, 224);

      // loading serialized torch script module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      // module = Module.load(assetFilePath(this, "ResNet34_15Apr2am.pt"));
      // module = Module.load(assetFilePath(this, "model.pt"));

      runModelOnTestSet();
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }

    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);

    // preparing input tensor
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

    // running the model
    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

    // getting tensor content as java array of floats
    float[] scores = outputTensor.getDataAsFloatArray();
    printArray(scores);
    scores = softmax(scores);
    printArray(scores);
    // searching for the index with maximum score
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxScoreIdx = i;
      }
    }

    String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

    // showing className on UI
    TextView textView = findViewById(R.id.text);
    textView.setText(className);
  }

  //#region Image-Processing

  // resizes bitmap to given dimensions
  public int[] getResizeTarget(Bitmap bm, int resizeHeight, int resizeWidth){
    float ratio, orig_height, orig_width;
    orig_height = bm.getHeight();
    orig_width = bm.getWidth();
    ratio = Math.min(orig_height/resizeHeight , orig_width/resizeWidth);
    int[] result = {Math.round(orig_height/ratio), Math.round(orig_width/ratio)};
    return result;
  }

  public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
    int width = bm.getWidth();
    int height = bm.getHeight();
    float scaleWidth = ((float) newWidth) / width;
    float scaleHeight = ((float) newHeight) / height;
    Matrix matrix = new Matrix();
    matrix.postScale(scaleWidth, scaleHeight);
    Bitmap resizedBitmap = Bitmap.createBitmap(
            bm, 0, 0, width, height, matrix, false);
    return resizedBitmap;
  }

  public Bitmap cropBitMap(Bitmap bm, int newWidth, int newHeight){
    int height = bm.getHeight();
    int width = bm.getWidth();
    int startRow = (int) Math.round((height-newHeight)*0.5);
    int startCol = (int) Math.round((width-newWidth)*0.5);
    Matrix matrix = new Matrix();
    Bitmap croppedBitmap = Bitmap.createBitmap(
            bm, startCol, startRow, newWidth, newHeight, matrix, false);
    return croppedBitmap;
  }

  //#endregion

  //#region PERMISSIONS HANDLING

  /** Method to check whether external media available and writable. This is adapted from
   http://developer.android.com/guide/topics/data/data-storage.html#filesExternal */

  private void get_permissions(){
    // request permission to write data (aka images) to the user's external storage of their phone
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
            && ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
              REQUEST_PERMISSION);
    }
  }

  // checks that the user has allowed all the required permission of read and write and camera. If not, notify the user and close the application
  @Override
  public void onRequestPermissionsResult(final int requestCode, @NonNull final String[] permissions, @NonNull final int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == REQUEST_PERMISSION) {
      if (!(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
        Toast.makeText(getApplicationContext(),"This application needs write permissions to run. Application now closing.",Toast.LENGTH_LONG);
        System.exit(0);
      }
    }
  }

  //#endregion

  //#region IO

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  private  void writeArrayToFile(String[] array, String fileName){
    File root = android.os.Environment.getExternalStorageDirectory();
    // See http://stackoverflow.com/questions/3551821/android-write-to-sd-card-folder

    File dir = new File (root.getAbsolutePath() + "/download");
    dir.mkdirs();
    File file = new File(dir, fileName);

    try {
      FileOutputStream f = new FileOutputStream(file);
      PrintWriter pw = new PrintWriter(f);
      for(String str: array){
        pw.println(str);
      }
      pw.flush();
      pw.close();
      f.close();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      Log.i("MEDIA", "******* File not found. Did you" +
              " add a WRITE_EXTERNAL_STORAGE permission to the   manifest?");
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  void printArray(float[] arr){
    for(int i=0; i<arr.length; i++){
      if(i == arr.length - 1){
        System.out.println(arr[i]);
      }
      else{
        System.out.print(arr[i]);
        System.out.print(" , ");
      }
    }
  }

  //#endregion

  //#region EXPERIMENTS

  public void runModelOnTestSet() throws IOException {
    Bitmap bitmap = null;
    Bitmap bitmap_orig = null;
    Module module = Module.load(assetFilePath(this, "ResNet34_15Apr2am.pt"));
    int TP_count = 0;
    final AssetManager assets = getAssets();
    String root_dir = "distracted_imgs_12";
    int count = 0;
    String[] test_sets = new String[0];
    // Get list of all the images in the folder
    try {
      test_sets = assets.list(root_dir);
    } catch (IOException e) {
      e.printStackTrace();
    }
    String[] results = new String[test_sets.length];

    for (String s : test_sets) {
      // Image Loading and processing
      bitmap_orig = BitmapFactory.decodeStream(getAssets().open(root_dir + "/" + s));
      int[] target_shape = getResizeTarget(bitmap_orig, 224, 224);
      int target_height = target_shape[0];
      int target_width = target_shape[1];
      // bitmap = getResizedBitmap(bitmap_orig,  target_width, target_height);
      bitmap = Bitmap.createScaledBitmap(bitmap_orig,  target_width, target_height, true);
      bitmap = cropBitMap(bitmap, 224, 224);
      final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
              TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

      // running the model
      final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

      // getting tensor content as java array of floats
      float[] scores = outputTensor.getDataAsFloatArray();
      if(scores[1] > scores[0]){
        TP_count += 1;
      }
      results[count] = s + ',' + Arrays.toString(scores);
      results[count] = results[count].replace("[","").replace("]","");
      //printArray(scores);
      //System.out.println(s);
      count += 1;
    }
    System.out.println("Out of 187, Total correct distraction detected = " + TP_count);
    writeArrayToFile(results, "results.txt");
  }

  public float[] softmax(float[] scores){
    float[] exp_scores = new float[scores.length];
    float[] result = new float[scores.length];
    float sum = 0;
    // Convert to exponential and computing the sum
    for(int i=0; i<scores.length; i++){
      exp_scores[i] = (float) Math.exp(scores[i]);
      sum = sum + exp_scores[i];
    }
    // divide exp scores with sum to get result
    for(int i=0; i<scores.length; i++){
      result[i] = exp_scores[i]/sum;
    }
    return result;
  }

  //#endregion
}
