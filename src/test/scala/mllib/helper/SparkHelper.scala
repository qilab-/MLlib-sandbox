package mllib.helper

import org.apache.spark.SparkContext

trait SparkHelper {

  def withSparkContext(test: SparkContext => Any) {
    test(SparkHelper.sc)
  }

}

object SparkHelper {

  private val sc = new SparkContext("local", "MLlib-sandbox Test")
  sc.setLogLevel("WARN")

  override protected def finalize = {
    sc.stop()
  }

}
