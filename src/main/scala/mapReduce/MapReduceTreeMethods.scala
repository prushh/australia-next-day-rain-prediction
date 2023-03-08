package it.unibo.andrp
package mapReduce

trait MapReduceTreeMethods {
  // Utility function to calculate accuracy of predictions
  def calculateAccuracy(predictions: Seq[Double], labels: Seq[Double]): Double = {
    val correctCount = predictions.zip(labels).count { case (predicted, actual) => predicted == actual }
    correctCount.toDouble / labels.size
  }
  def train(train_data:Seq[DataPoint]):Unit
  def predict(point:DataPoint):Double
  def accuracy(test_data:Seq[DataPoint]):Double
}
