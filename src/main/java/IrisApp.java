import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

public class IrisApp {
	
	public static void main(String[] args) {
	
	MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
	.list()
	.build();
	}
}
