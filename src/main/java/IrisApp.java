import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.learning.config.Adam;

public class IrisApp {
	
	public static void main(String[] args) {
	double learninRate=0.001;
	MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
	.updater(new Adam(learninRate))
			
			
	.list()
	.build();
	}
}
