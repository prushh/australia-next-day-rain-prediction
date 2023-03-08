package it.unibo.andrp
package mapReduce
import scala.math._

import org.apache.spark.SparkContext

class GradientBoostingMapReduce(sc: SparkContext, numCluster: Int = 5, numTrees: Int = 5, featureSubsetStrategy: String = "all", minSplitSize: Int = 5, impurityMeasure: String = "gini") extends MapReduceTreeMethods {
  val fraction: Double = 1.0 / numCluster
  var gbc = List.fill(numCluster)(new GradientBoostingClassifier(maxDepth = 3,numIterations = 20,featureSubsetStrategy=featureSubsetStrategy))

  override def train(train_data: Seq[DataPoint]): Unit = {
    val data_mapred = sc.parallelize(train_data)

    // Creazione degli alberi
    gbc = gbc.map { rf =>
      val sample = data_mapred.sample(true, fraction)
      rf.train(sample.collect().toList)
      rf
    }
  }

  // Funzione di predizione
  override def predict(point: DataPoint): Double = {
    val predictions = gbc.map(rf => round(rf.predict(point)))
    predictions.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
  }

  override def accuracy(test_data: Seq[DataPoint]): Double = {
    //Make predictions on the testing data
    val predictions_map = test_data.map(predict)

    //Evaluate the accuracy of the predictions
    calculateAccuracy(predictions_map, test_data.map(_.label))
  }



}
