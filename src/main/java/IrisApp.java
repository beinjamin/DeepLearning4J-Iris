import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;

public class IrisApp {
	
	public static void main(String[] args) {
	double learninRate=0.001;
	int numInputs=4;
	int numHidden=10;
	int numOutputs=3;
	MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
	.updater(new Adam(learninRate))
			
			
	.list()
	 .layer(0 , new DenseLayer.Builder()
			 .nIn(numInputs)
			 .nOut(numHidden)
			 .activation(Activation.SIGMOID)
			 .build()
			 )
	 .layer(1, new DenseLayer.Builder()
			 .nIn(numHidden)
			 .nOut(numOutputs)
			 .activation(Activation.SOFTMAX)
			 .lossFunction(LossFuntions.LossFuntion.ME)
			 
			 )
	.build();
	}
}
