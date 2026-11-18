import ij.IJ;
import ij.ImagePlus;
import ij.Prefs;
import ij.io.FileSaver;
import ij.io.Opener;
import ij.process.ColorProcessor;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;
import java.io.File;
import java.awt.Color;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Test-only driver for the pinned upstream producer; algorithms remain upstream methods. */
public final class RunProducer {
    private static final int WIDTH = 1210;
    private static final int HEIGHT = 566;
    private static final int RADIUS = 20;

    private RunProducer() {}

    private static ImagePlus canvas(int width, int height, String kind) {
        ColorProcessor pixels = new ColorProcessor(width, height);
        if (kind.equals("full")) {
            pixels.setColor(Color.WHITE);
            pixels.fill();
        } else if (kind.equals("isolated")) {
            pixels.set(width / 2, height / 2, 0xFFFFFFFF);
        } else if (kind.equals("border")) {
            pixels.set(600, 0, 0xFFFFFFFF);
            pixels.set(600, height - 1, 0xFFFFFFFF);
        } else if (kind.equals("colors")) {
            pixels.set(360, 280, 0xFFFF0000);
            pixels.set(460, 280, 0xFF00FF00);
            pixels.set(560, 280, 0xFF0000FF);
            pixels.set(660, 280, 0xFFFFFFFF);
            pixels.set(760, 280, 0xFF010000);
            pixels.set(860, 280, 0xFF020000);
        } else if (kind.equals("labels")) {
            pixels.set(100, 50, 0xFFFFFFFF);
            pixels.set(width - 100, 50, 0xFFFFFFFF);
            pixels.set(200, 104, 0xFFFFFFFF);
            pixels.set(width - 150, 99, 0xFFFFFFFF);
            pixels.set(400, 105, 0xFFFFFFFF);
            pixels.set(605, 283, 0xFFFFFFFF);
        }
        return new ImagePlus(kind, pixels);
    }

    private static void saveTiff(ImagePlus image, Path path) throws Exception {
        if (!new FileSaver(image).saveAsTiff(path.toString())) {
            throw new IllegalStateException("Could not write " + path);
        }
        image.close();
    }

    private static void runOriginal(Path inputDir, Path outputDir) {
        Gradient_conv_multithread producer = new Gradient_conv_multithread();
        producer.thread_num_ = 1;
        producer.maxradius = RADIUS;
        producer.Threfunction(inputDir.toString() + File.separator,
                inputDir.toFile().listFiles(), outputDir.toString() + File.separator,
                1, RADIUS);
    }

    private static void runWithoutLabelClearing(Path input, Path output) throws Exception {
        ImagePlus image = new Opener().openImage(input.toString());
        if (image == null) throw new IllegalArgumentException("Could not open " + input);
        ImageConverter converter = new ImageConverter(image);
        converter.convertToGray8();
        IJ.run(image, "Maximum...", "radius=" + RADIUS);
        converter = new ImageConverter(image);
        converter.convertToGray16();
        ImageProcessor ip = image.getProcessor();
        for (int i = 0; i < ip.getPixelCount(); i++) {
            ip.set(i, ip.get(i) < 1 ? 65535 : 0);
        }
        Gradient_conv_multithread producer = new Gradient_conv_multithread();
        ImageProcessor result = producer.gradientslice(ip);
        image.setProcessor(result);
        producer.writeImage(image, output.toString());
        image.close();
    }

    public static void main(String[] args) throws Exception {
        if (!"1.53v".equals(IJ.getVersion())) {
            throw new IllegalStateException("Expected ImageJ 1.53v, got " + IJ.getVersion());
        }
        Path fixtureInputs = Path.of(args[0]);
        Path work = Path.of(args[1]);
        Path outputs = Path.of(args[2]);
        Path vncInputs = Path.of(args[3]);
        Path referenceInputs = work.resolve("reference-inputs");
        Path referenceOutputs = work.resolve("reference-outputs");
        Path vncReferenceInputs = work.resolve("vnc-reference-inputs");
        Path vncReferenceOutputs = work.resolve("vnc-reference-outputs");
        Path syntheticInputs = work.resolve("synthetic-inputs");
        Path syntheticOutputs = work.resolve("synthetic-outputs");
        Files.createDirectories(referenceInputs);
        Files.createDirectories(referenceOutputs);
        Files.createDirectories(vncReferenceInputs);
        Files.createDirectories(vncReferenceOutputs);
        Files.createDirectories(syntheticInputs);
        Files.createDirectories(syntheticOutputs);
        Files.createDirectories(outputs);

        // Lock process-global conversion settings instead of inheriting user preferences.
        ColorProcessor.setWeightingFactors(1d / 3d, 1d / 3d, 1d / 3d);
        ImageConverter.setDoScaling(true);
        Prefs.weightedColor = false;
        Prefs.useInvertingLut = false;

        try (var files = Files.list(fixtureInputs)) {
            for (Path source : files.filter(p -> p.toString().endsWith(".tif")).toList()) {
                Files.copy(source, referenceInputs.resolve(source.getFileName()),
                        StandardCopyOption.REPLACE_EXISTING);
            }
        }
        runOriginal(referenceInputs, referenceOutputs);
        try (var files = Files.list(referenceOutputs)) {
            for (Path generated : files.filter(p -> p.toString().endsWith(".png")).toList()) {
                Files.copy(generated, outputs.resolve(generated.getFileName()),
                        StandardCopyOption.REPLACE_EXISTING);
            }
        }

        // Retained real VNC (MANC) TIFFs run through the identical upstream path.
        try (var files = Files.list(vncInputs)) {
            for (Path source : files.filter(p -> p.toString().endsWith(".tif")).toList()) {
                Files.copy(source, vncReferenceInputs.resolve(source.getFileName()),
                        StandardCopyOption.REPLACE_EXISTING);
            }
        }
        runOriginal(vncReferenceInputs, vncReferenceOutputs);
        try (var files = Files.list(vncReferenceOutputs)) {
            for (Path generated : files.filter(p -> p.toString().endsWith(".png")).toList()) {
                Files.copy(generated, outputs.resolve(generated.getFileName()),
                        StandardCopyOption.REPLACE_EXISTING);
            }
        }

        String[] kinds = {"empty", "full", "isolated", "border", "colors", "labels"};
        for (String kind : kinds) {
            saveTiff(canvas(WIDTH, HEIGHT, kind), syntheticInputs.resolve(kind + ".tif"));
        }
        saveTiff(canvas(573, 1209, "isolated"), syntheticInputs.resolve("vnc-isolated.tif"));
        runOriginal(syntheticInputs, syntheticOutputs);
        try (var files = Files.list(syntheticOutputs)) {
            for (Path generated : files.filter(p -> p.toString().endsWith(".png")).toList()) {
                String name = generated.getFileName().toString();
                String id = name.equals("vnc-isolated.png") ? "vnc-canvas.png" : "synthetic-" + name;
                Files.copy(generated, outputs.resolve(id), StandardCopyOption.REPLACE_EXISTING);
            }
        }
        runWithoutLabelClearing(syntheticInputs.resolve("labels.tif"),
                outputs.resolve("synthetic-labels-no-labels.png"));

        // Prevent ImageJ's AWT event queue from keeping the batch process alive.
        System.exit(0);
    }
}
