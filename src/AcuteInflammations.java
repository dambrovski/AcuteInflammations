
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class AcuteInflammations {
	

	public static void BuildModel() throws Exception {
		
		ArffLoader arffLoader = new ArffLoader();
	    arffLoader.setSource(new File("urinary_bladder.arff"));
	   
	    Instances data = arffLoader.getDataSet();
	    
		data.setClassIndex(data.numAttributes()-1); //define o último atributo como a classe
		//Constroi o classificador
		
		//-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1
		String [] options =  {"-P", "100", "-I", "100", "-num-slots", "1", "-K", "0", "-M", "1.0", "-V", "0.001", "-S" ,"1"};
		RandomForest rf = new RandomForest();
		rf.setOptions(options);
		rf.buildClassifier(data);
		//Salva modelo
		weka.core.SerializationHelper.write("../urinary_bladder.model", rf); 
	}
	
	public static void Evaluate() throws FileNotFoundException, Exception {
		
		//Carrega os dados do csv e joga no objeto (Instances)
		DataSource source = new DataSource("urinary_bladder.arff");
		Instances data = source.getDataSet();
		Instances testset = data;
		data.setClassIndex(data.numAttributes()-1); //define o último attribute como classe

		//Abre o modelo que está no disco
		RandomForest rf = (RandomForest) SerializationHelper.read(new FileInputStream("../urinary_bladder.model"));
		Evaluation eval = new Evaluation(testset);
		Random rand = new Random(1);  // seed = 1
		int folds = 10;
		eval.crossValidateModel(rf, testset, folds, rand);
		
		//Mostra os resultados dos testes
		System.out.println(eval.toSummaryString());
		System.out.println("-----------------------");
		System.out.println(eval.toClassDetailsString());
		System.out.println("-----------------------");		
		System.out.println(eval.toMatrixString());
	}
	
	public static void Classify() throws FileNotFoundException, Exception {
		
		RandomForest rf = (RandomForest) SerializationHelper.read(new FileInputStream("../urinary_bladder.model"));
		ArrayList<String> ClassVal = new ArrayList<String>();
		ClassVal.add("no");
		ClassVal.add("yes");
		DataSource ds = new DataSource("urinary_bladder.arff");
		Instances ins = ds.getDataSet();
		ins.setClassIndex(ins.numAttributes()-1);
		
		//Prepara a instancia que será classificada
		Instance newInstance= new DenseInstance(10);
		newInstance.setDataset(ins);
		
		Scanner sc = new Scanner(System.in);
		
		System.out.println("Temperatura do Paciente : ");
		String v1_string = sc.nextLine();
		System.out.println("Apresenta Nausea?: ");
		String v2 = sc.nextLine();
		System.out.println("Dores na lombar?: ");
		String v3 = sc.nextLine();
		System.out.println("Necessidade continua de urinar?: ");
		String v4 = sc.nextLine();
		System.out.println("Dor na micção?: ");
		String v5 = sc.nextLine();
		System.out.println("Queimação na uretra?: ");
		String v6 = sc.nextLine();
		
		float v1 = Float.parseFloat(v1_string);
		// Instancias
		newInstance.setValue(0, v1);
		newInstance.setValue(1, v2);
		newInstance.setValue(2, v3);
		newInstance.setValue(3, v4);
		newInstance.setValue(4, v5);
		newInstance.setValue(5, v6);
		
        System.out.println(ClassVal.get((int) rf.classifyInstance(newInstance)));
		double LikelihoodDistribution[] = rf.distributionForInstance(newInstance);
		for(int i=0;i<LikelihoodDistribution.length;i++) {
			System.out.println(ClassVal.get(i) + ":" + LikelihoodDistribution[i]);
		}
		
		sc.close();
	}
	
	
	public static void main (String args[]) {
	
		try {
			BuildModel();
			Evaluate();
			Classify();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
