/**
  * Created by root on 20/04/2017.
  */
import java.io.File
import java.util.Random

import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, ImagePreProcessingScaler}
object Pretrained {
  protected val height = 100
  protected val width = 100
  protected val channels = 3
  protected val numExamples = 80
  protected val numLabels = 4
  protected val batchSize = 20

  def main(args:Array[String])={
    val scaler = new ImagePreProcessingScaler(0, 1)

    val locationToSave = new File("/home/austin/Documents/model.bin")
    val model = ModelSerializer.restoreMultiLayerNetwork(locationToSave)

    val mainPath = new File("/home/austin/Documents/test/test.jpg")
   // val labelMaker = new ParentPathLabelGenerator
//    val filesInDir = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, new Random(123))
//    import org.datavec.api.io.filters.BalancedPathFilter
//    val pathFilter = new BalancedPathFilter(new Random(123), NativeImageLoader.ALLOWED_FORMATS)
//    val filesInDirSplit = filesInDir.sample(pathFilter,1.0)
//    val trainData = filesInDirSplit(0)
//    val recordReader = new ImageRecordReader(height, width, channels)
//
//    recordReader.initialize(trainData)
//    var dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
//    scaler.fit(dataIter)
//    dataIter.setPreProcessor(scaler)
//    val test:DataSet = dataIter.next()
//    val expectedResult = test.getLabelName(0)
//    val res=model.predict(test)
//    println("Supposed to be "+expectedResult)
//    println("It predicted as "+res)
    import org.datavec.image.loader.NativeImageLoader
    import org.nd4j.linalg.api.ndarray.INDArray
    val loader = new NativeImageLoader(100, 100, 3)
    val image = loader.asMatrix(mainPath)
    val res = model.predict(image)
    val out2 = model.output(image)
      println(out2.getFloat(0))
      println(out2.getColumn(0))
      println(out2.getInt(0))


    println(res(0))

  }
}
