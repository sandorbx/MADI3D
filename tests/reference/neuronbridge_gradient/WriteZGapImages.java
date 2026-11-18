package org.janelia.colormipsearch.cds;

import ij.ImagePlus;
import ij.io.FileSaver;
import ij.io.Opener;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.janelia.colormipsearch.ImageTestUtils;
import org.janelia.colormipsearch.imageprocessing.ColorTransformation;
import org.janelia.colormipsearch.imageprocessing.ImageArray;
import org.janelia.colormipsearch.imageprocessing.ImageArrayUtils;
import org.janelia.colormipsearch.imageprocessing.ImageProcessing;
import org.janelia.colormipsearch.imageprocessing.ImageTransformation;
import org.janelia.colormipsearch.imageprocessing.LImageUtils;

/** Writes the original test's derived Z-gap supplier output without changing its pixels. */
public final class WriteZGapImages {
    private WriteZGapImages() {}

    public static void main(String[] args) throws Exception {
        Path inputRoot = Path.of(args[0]);
        Path casesPath = Path.of(args[1]);
        Path outputDir = Path.of(args[2]);
        Files.createDirectories(outputDir);
        List<Map<String, String>> cases = RunReference.JSON.readValue(casesPath.toFile(), List.class);
        for (Map<String, String> c : cases) {
            if (c.get("zgap") != null) continue;
            Path source = inputRoot.resolve(c.get("lm"));
            Path output = outputDir.resolve(Path.of(c.get("lm")).getFileName().toString().replaceFirst("\\.tif$", ".png"));
            ImagePlus imp = new Opener().openImage(source.toString());
            ImageArray<?> lm = ImageArrayUtils.fromImagePlus(imp);
            for (boolean noLabels : new boolean[]{false, true}) {
                Path variant = noLabels
                        ? output.resolveSibling(output.getFileName().toString().replaceFirst("\\.png$", "-no-labels.png"))
                        : output;
                ImageTransformation clear = noLabels ? ImageTransformation.IDENTITY
                        : ImageTransformation.clearRegion(ImageTestUtils.getExcludedRegions().getRegion(lm));
                ImageArray<?> zgap = ImageProcessing.create(clear)
                        .applyColorTransformation(ColorTransformation.mask(20))
                        .unsafeMaxFilter(10)
                        .applyTo(LImageUtils.create(lm))
                        .toImageArray();
                ImagePlus result = new ImagePlus("zgap", ImageArrayUtils.toImageProcessor(zgap));
                if (!new FileSaver(result).saveAsPng(variant.toString())) {
                    throw new IllegalStateException("Could not write " + variant);
                }
                result.close();
            }
            imp.close();
        }
    }
}
