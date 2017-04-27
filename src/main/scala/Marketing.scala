/**
  * Created by Austin on 19/04/2017.
  */
import java.io.File
import java.util

import org.apache.spark.SparkConf
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.datavec.api.split.{FileSplit, ListStringSplit}
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.schema.Schema
import org.datavec.api.writable.Writable
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.{StringToWritablesFunction, WritablesToStringFunction}
import java.util.{Date, List, Random}

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, OutputLayer, RBM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader
import org.datavec.api.records.reader.impl.csv
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j



object Marketing {
  val numLinesToSkip = 0
  val delimiter = ","

  def main(args:Array[String]) = {
    // Load CSV
    // TODO: You should modify this path to your own directory
    var list =new util.ArrayList(loadCSV("/home/austin/Documents/marketing","bank.csv"))
    // I do not want the header row so i ge the sublist
    list = new util.ArrayList(list.subList(1,list.size()-1))
    // ArrayList -> ArrayList[ArrayList[String]]
    var list2 = new util.ArrayList[List[String]]()
    val seed = System.nanoTime
    for(i<-0 until list.size()){
      val array = list.get(i).split(",")
      val list3 = new util.ArrayList[String]()
      for(j<-0 until array.length){
        list3.add(array(j))
      }
      list2.add(list3)
    }
    // The recorder
    val trainRecorder =new ListStringRecordReader()
    trainRecorder.initialize(new ListStringSplit(list2))
    // iterator having the batch size of 100
    // target variable at 11th columns(start from 0)
    // the target has two possibilities -> 0, 1
    val trainIterator = new RecordReaderDataSetIterator(trainRecorder,100,11,2)
    // this one for test using batch size of 20
    val testIterator = new RecordReaderDataSetIterator(trainRecorder,20,11,2)
    val featuresTrain = new util.ArrayList[INDArray]()
    val featuresTest = new util.ArrayList[INDArray]()
    val labelsTest = new util.ArrayList[INDArray]()
    /***
      * This is just to split the dataset randomly
      */
    // Random with 12345 seeding
    val r = new Random(12345)
    while(trainIterator.hasNext()){
      val ds = trainIterator.next()
      // Split training and testing to 80:20
      val split = ds.splitTestAndTrain(80,r)
      featuresTrain.add(split.getTrain().getFeatureMatrix)
      val dsTest = split.getTest
      featuresTest.add(dsTest.getFeatureMatrix)
      val indexesTrain = Nd4j.argMax(split.getTrain.getLabels,1)
      labelsTest.add(indexesTrain)
      val indexesTest = Nd4j.argMax(dsTest.getLabels,1)
      labelsTest.add(indexesTest)
    }
    /**
      * Neural network configuration
      * TODO: Modify the configuration
      * Q1: Is using RBM a good choice
      * Q2: Is this the right way of using RBM? (You play and you know, if you get the concept right, you might able to do it)
      */

    val conf = new NeuralNetConfiguration.Builder()
      .seed(123)    //Random number generator seed for improved repeatability. Optional.
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .learningRate(0.05)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
      .gradientNormalizationThreshold(0.5)
      .list()
      .layer(0, new RBM.Builder().nIn(11).nOut(15).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
      .layer(1, new RBM.Builder().nIn(15).nOut(8).lossFunction(LossFunctions.LossFunction.MSE).build())
      .layer(2, new RBM.Builder().nIn(8).nOut(15).lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build())
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(15).nOut(2).build())
      .pretrain(true).backprop(true)
      .build()

    // Finish configuration
    val net = new MultiLayerNetwork(conf)
    net.init()
    // 1 Epoch
    val nEpochs = 1
    // Loop the epoch
    for (i<- 0 until nEpochs) {
      for(j<-0 until featuresTrain.size()){
        // Training happen
        net.fit(featuresTrain.get(j))
      }
      // Evaluation
      val evaluation = net.evaluate(testIterator)


      /**
        * Showing the evaluation statistics and result
        */
      println(evaluation.stats())
      println(evaluation.getConfusionMatrix)
      println("Epoch "+i+": Testing")
      for(j<-0 until featuresTest.size()){
        val result = net.predict(featuresTest.get(j))
        for(k<-0 until result.length){
          println("Result for "+featuresTest.get(j).getRow(k)+": "+result(k))
        }
      }
    }

  }

  def loadCSV(path:String,fileName: String) :util.List[String]= {
    val inputPath = path+fileName
    val timeStamp = String.valueOf(new Date().getTime)
    val outputPath = path+"report_processed_"+timeStamp
    val baseTrainDir = new File(path,"train")
    /**
      * The schema
      */
    val inputDataSchema:Schema = new Schema.Builder()
      .addColumnInteger("age")
      .addColumnString("job")
      .addColumnInteger("marital")
      .addColumnInteger("education")
      .addColumnString("housing")
      .addColumnInteger("loan")
      .addColumnInteger("duration")
      .addColumnInteger("campaign")
      .addColumnInteger("pdays")
      .addColumnInteger("previous")
      .addColumnDouble("emp.var.rate")
      .addColumnDouble("nr.employed")
      .addColumnInteger("subscribe")
      .build()

    val tp = new TransformProcess.Builder(inputDataSchema)
      .removeColumns("job")
      .build()

    val numActions = tp.getActionList().size()
    for(i<- 0 until numActions){
      tp.getActionList.get(i)
      println(tp.getSchemaAfterStep(i))
    }

    val sparkConf = new SparkConf()
    sparkConf.setMaster("local[*]")
    sparkConf.setAppName("Predict Marketing")
    val sc = new JavaSparkContext(sparkConf)

    val lines = sc.textFile(path)
    val writable:JavaRDD[java.util.List[Writable]] = lines.map(new StringToWritablesFunction(new csv.CSVRecordReader()))
    val processed = SparkTransformExecutor.execute(writable,tp)

    val toSave = processed.map(new WritablesToStringFunction(","))
    toSave.collect()
  }

}