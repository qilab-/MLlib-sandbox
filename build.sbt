name := "MLlib-sandbox"

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "1.4.0",
  "org.apache.spark" % "spark-mllib_2.11" % "1.4.0",
  // "com.github.wookietreiber" % "scala-chart_2.11" % "0.4.2",
  "org.scalatest" %% "scalatest" % "2.2.4" % "test"
)

