name := "NerutiDeepLearningExample"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.deeplearning4j" % "dl4j-spark_2.11" % "0.8.0_spark_2"
libraryDependencies += "com.beust" % "jcommander" % "1.48"
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.8.0"
libraryDependencies += "org.datavec" % "datavec-api" % "0.8.0"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-ui_2.11" % "0.8.0"

resolvers ++= Seq(
  "Typesafe" at "http://repo.typesafe.com/typesafe/releases/",
  "Java.net Maven2 Repository" at "http://download.java.net/maven/2/"
)