import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by alse on 8/30/17.
 */
public class Inception {
    private static final Logger log = LoggerFactory.getLogger(Inception.class);
    private byte[] graphDef;
    private List<String> labels;
    private Output output;
    private Session s;

    public Inception(String graphPath, String labelsPath) throws IOException{
        graphDef = Files.readAllBytes(Paths.get(graphPath));
        labels = Files.readAllLines(Paths.get(labelsPath));

        Graph g = new Graph();
        s = new Session(g);
        GraphBuilder b = new GraphBuilder(g);
        // - The model was trained with images scaled to 224x224 pixels.
        // - The colors, represented as R, G, B in 1-byte each were converted to
        //   float using (value - Mean)/Scale.
        final int H = 224;
        final int W = 224;
        final float mean = 117f;
        final float scale = 1f;
        output = b.div(
                    b.sub(
                        b.resizeBilinear(
                                b.expandDims(
                                        b.cast(b.decodeJpeg(b.placeholder("input", DataType.STRING), 3), DataType.FLOAT),
                                        b.constant("make_batch", 0)),
                                b.constant("size", new int[] {H, W})),
                        b.constant("mean", mean)),
                    b.constant("scale", scale));
    }

    public float[] predict(String imageFilePath) throws IOException{
        byte[] imageBytes = Files.readAllBytes(Paths.get(imageFilePath));
        return predict(imageBytes);
    }

    public float[] predict(byte[] imageBytes) {
        final Tensor input = Tensor.create(imageBytes);
        Tensor image = s.runner().feed("input", input).fetch(output.op().name()).run().get(0);
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 Tensor result = s.runner().feed("input", image).fetch("output").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            }
        }
    }

    public int bestMatchIndex(String imageFilePath) throws IOException {
        byte[] imageBytes = Files.readAllBytes(Paths.get(imageFilePath));
        return bestMatchIndex(imageBytes);
    }

    public int bestMatchIndex(byte[] imageBytes) throws IOException {
        float[] probabilities = predict(imageBytes);
        int best = 0;
        for (int i = 0; i < probabilities.length; i++) {
            best = probabilities[i] > probabilities[best] ? i : best;
        }
        return best;
    }

    public List<String> getLabels() {
        return labels;
    }
}
