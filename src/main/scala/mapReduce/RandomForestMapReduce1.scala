package it.unibo.andrp
package mapReduce

import org.apache.spark.SparkContext

class RandomForestMapReduce1(sc: SparkContext, numCluster: Int = 5, numTrees: Int = 5, featureSubsetStrategy: String = "all", minSplitSize: Int = 5, impurityMeasure: String = "gini") extends MapReduceTreeMethods {
  val fraction: Double = 1.0 / numCluster
  var rfs = List.fill(numCluster)(new RandomForest(maxDepth = 3, numTrees = numTrees))

  override def train(train_data: Seq[DataPoint]): Unit = {
    val data_mapred = sc.parallelize(train_data)

    // Creazione degli alberi
    rfs = rfs.map { rf =>
      val sample = data_mapred.sample(true, fraction)
      rf.train(sample)
      rf
    }
  }

  // Funzione di predizione
  override def predict(point: DataPoint): Double = {
    val predictions = rfs.map(rf => rf.predict(point))
    predictions.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
  }

  override def accuracy(test_data: Seq[DataPoint]): Double = {
    //Make predictions on the testing data
    val predictions_map = test_data.map(predict)

    //Evaluate the accuracy of the predictions
    calculateAccuracy(predictions_map, test_data.map(_.label))
  }

}
