import java.io.IOException;

/**
 * Created by alse on 8/30/17.
 */
public class InceptionTest {
    public static void main(String[] args) {
        ClassLoader classLoader = ClassLoader.getSystemClassLoader();
        try {
            Inception model = new Inception(classLoader.getResource("tensorflow_inception_graph.pb").getPath(),
                    classLoader.getResource("imagenet_comp_graph_label_strings.txt").getPath());
            System.out.println(model.getLabels().get(model.bestMatchIndex(classLoader.getResource("dog.jpg").getPath())));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
