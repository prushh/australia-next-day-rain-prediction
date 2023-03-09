package it.unibo.andrp
package mapReduce
import scala.math._

import org.apache.spark.SparkContext

/**
 * Parallel implementation of the gradient boosting algorithm using Spark's MapReduce framework.
 * @param sc the SparkContext
 * @param numCluster the number of clusters to use for parallelizing the GBC
 * @param numIterations the number of iterations to perform for training the GBC
 * @param maxDepth the maximum depth of the decision trees in the GBC
 * @param minSplitSize the minimum number of instances required to split an internal node in the GBC
 * @param featureSubsetStrategy the strategy for selecting features in each decision tree in the GBC
 * @param impurityMeasure the impurity measure to use for splitting nodes in the decision trees in the GBC
 * @param gamma the regularization parameter for the GBC
 * */
class GradientBoostingMapReduce(sc: SparkContext, numCluster: Int = 5, numIterations: Int = 20, learningRate: Double = 0.1, maxDepth : Int = 3, minSplitSize: Int = 10, featureSubsetStrategy: String = "all", impurityMeasure: String = "gini", gamma: Int = 1) extends MapReduceTreeMethods {
  /**
   * The fraction of data to use for each cluster.
   */
  val fraction: Double = 1.0 / numCluster

  /**
   * A list of Gradient Boosting Classifiers, one for each cluster.
   */
  var gbc: List[GradientBoostingClassifier] = List.fill(numCluster)(new GradientBoostingClassifier(maxDepth = maxDepth, numIterations = numIterations, learningRate = learningRate, featureSubsetStrategy = featureSubsetStrategy, minSplitSize = minSplitSize, impurityFunc = impurityMeasure, gamma = gamma))

  /**
   * Trains the GBD on the given dataset.
   *
   * @param train_data the training dataset
   */
  override def train(train_data: Seq[DataPoint]): Unit = {
    // Create an RDD from the training data using SparkContext
    val data_mapred = sc.parallelize(train_data)
    var count = 0;

    // Train each GBD on a fraction of the data
    gbc = gbc.map { rf =>
      println(s"$count CLUSTER")
      count = count + 1;
      val sample = data_mapred.sample(true, fraction)
      rf.train(sample.collect().toList)
      rf
    }
  }

  /**
   * Predicts the label of the given data point using the trained GBD
   *
   * @param point the data point to predict
   * @return the predicted label
   */
  override def predict(point: DataPoint): Double = {
    // Predicting by aggregating the predictions of all the classifiers in parallel
    val predictions = gbc.map(rf => round(rf.predict(point)))

    // Finding the most frequent predicted label among all the classifiers
    predictions.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
  }

  /**
   * Calculates the accuracy of the model on the test data.
   *
   * @param test_data the test data to evaluate the accuracy of the model.
   * @return the accuracy of the model on the test data.
   */
  override def accuracy(test_data: Seq[DataPoint]): Double = {
    // Make predictions on the testing data
    val predictions_map = test_data.map(predict)

    // Evaluate the accuracy of the predictions
    calculateAccuracy(predictions_map, test_data.map(_.label))
  }

}
