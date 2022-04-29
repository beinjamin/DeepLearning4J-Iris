import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class IrisApp {
	
	public static void main(String[] args) throws Exception {
	double learninRate=0.001;
	int numInputs=4;
	int numHidden=10;
	int numOutputs=3;
	System.out.println("Creation du model");
	MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
			.seed(1234)
	.updater(new Adam(learninRate))
			
			
	.list()
	 .layer(0 , new DenseLayer.Builder()
			 .nIn(numInputs)
			 .nOut(numHidden)
			 .activation(Activation.SIGMOID)
			 .build()
			 )
	 .layer(1, new OutputLayer.Builder()
			 .nIn(numHidden)
			 .nOut(numOutputs)
			 .activation(Activation.SOFTMAX)
			 .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
			 .build()
			 )
	.build();
	MultiLayerNetwork model=new MultiLayerNetwork(configuration);
	model.init();
	System.out.println(configuration.toJson());

	System.out.print("Entrainement du model");
	File fileTrain=new ClassPathResource("iris-train.csv").getFile();
	
	
	RecordReader recordReaderTrain=new CSVRecordReader();	
	recordReaderTrain.initialize(new FileSplit(fileTrain));
	}
}
