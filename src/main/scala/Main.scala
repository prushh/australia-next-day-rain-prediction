package it.unibo.andrp

import config.SparkProjectConfig
import utils.{ReadCSV, TimestampFormatter}

import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

import scala.collection.immutable.Vector
import scala.io.Source
import scala.util.Using

@main def run(): Unit =
    /*
     * Loading Spark and Hadoop.
     */
    val sparkSession = SparkProjectConfig.sparkSession("local[*]", 1)
    val sparkContext = sparkSession.sparkContext

    val data = MLUtils.loadLibSVMFile(sparkContext, "data/weatherAUS-final.data")

    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)
    println(s"SVMWithSGD.weights: ${model.weights}")

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
        val score = model.predict(point.features)
        (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println(s"Area under ROC = $auROC")

    sparkSession.stop()

//    // initialize global support vector
//    val global_support_vector : Vector[Vector[Double]] = Vector.empty
//    /*val hyp : Vector[Double] => Double = _ => {
//
//    }*/
//
//    // load data
//    val (data_csv,target_csv) = ReadCSV("data/weatherAUS-final.csv")
//    val target_svm = target_csv.map {
//        case 0.0 => -1.0
//        case 1.0 => 1.0
//    }
//    //println(data_csv)
//    // divide data into chunks
//    /*val num_maps = 40
//    val chunkSize = data_csv.size / num_maps
//    val chunks = data_csv.grouped(chunkSize).toVector*/
//    val svm= new BinarySVM(data_csv,target_svm)
//
//    //println(svm.weights)
//    svm.fit()
//    println(svm.weights)
//    //print_graph()
//    println("Classification accuracy:" + (svm.predict(svm.x_bias), target_svm).zipped.count(i => i._1 == i._2).toDouble / svm.x_bias.length)
//    //println(svm.get_support_vectors())
//
//    // while hyp t =! hyp t-1
//
//        // for on chunks/computer
//
//            // split training and testing
//
//            // add global support vectors to training set
//
//            // train svm
//
//            // get new support vectors
//
//        // for on chunks/computer

