package org.janelia.colormipsearch.cds;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import ij.io.Opener;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;
import java.util.function.Supplier;
import org.janelia.colormipsearch.ImageTestUtils;
import org.janelia.colormipsearch.imageprocessing.*;
import org.janelia.colormipsearch.model.ComputeFileType;

/** Test-only input preparation and observation of the original API. No scoring implementation. */
public class RunReference {
    static final ObjectMapper JSON = new ObjectMapper();
    static final Map<String, Object> ASSETS = new LinkedHashMap<>();

    static String hex(byte[] bytes) {
        StringBuilder s = new StringBuilder();
        for (byte b : bytes) s.append(String.format("%02x", b & 255));
        return s.toString();
    }

    static ImageArray<?> load(Path root, String name) throws Exception {
        Path path = root.resolve(name);
        ImageArray<?> a = ImageArrayUtils.fromImagePlus(new Opener().openImage(path.toString()));
        if (!ASSETS.containsKey(name)) {
            boolean gray16 = name.startsWith("grad/");
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("file_sha256", hex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path))));
            m.put("decoded_sha256", pixelHash(a, gray16));
            m.put("shape", gray16 ? new int[]{a.getHeight(), a.getWidth()} : new int[]{a.getHeight(), a.getWidth(), 3});
            m.put("dtype", gray16 ? "<u2" : "|u1");
            m.put("pixel_encoding", gray16 ? "unsigned uint16 little endian, row-major" : "RGB uint8, row-major");
            ASSETS.put(name, m);
        }
        return a;
    }

    static String pixelHash(ImageArray<?> a, boolean gray16) throws Exception {
        MessageDigest pixels = MessageDigest.getInstance("SHA-256");
        for (int i = 0; i < a.getPixelCount(); i++) {
            int p = a.get(i);
            if (gray16) { pixels.update((byte)p); pixels.update((byte)(p >>> 8)); }
            else { pixels.update((byte)(p >>> 16)); pixels.update((byte)(p >>> 8)); pixels.update((byte)p); }
        }
        return hex(pixels.digest());
    }

    static ColorDepthSearchAlgorithm<ShapeMatchScore> direct(ImageArray<?> em) {
        ImageTransformation clear = ImageTransformation.clearRegion(ImageTestUtils.getExcludedRegions().getRegion(em));
        LImage query = LImageUtils.create(em, 0, 0, 0, 0).mapi(clear);
        LImage high = LImageUtils.combine2(query.mapi(ImageTransformation.unsafeMaxFilter(60)),
                query.mapi(ImageTransformation.unsafeMaxFilter(20)), (p1,p2) -> (p2 & 0xFFFFFF) != 0 ? 0xFF000000 : p1)
                .map(ColorTransformation.toGray16WithNoGammaCorrection())
                .map(ColorTransformation.gray8Or16ToSignal(0)).reduce();
        LImage mask = query.map(ColorTransformation.toGray16WithNoGammaCorrection())
                .map(ColorTransformation.gray8Or16ToSignal(2)).reduce();
        return new Shape2DMatchColorDepthSearchAlgorithm(query, mask, high, null, 20, true, clear);
    }

    public static void main(String[] args) throws Exception {
        Path root = new File(args[0]).toPath();
        List<Map<String, String>> cases = JSON.readValue(new File(args[1]), List.class);
        List<Object> rows = new ArrayList<>();
        ColorDepthSearchAlgorithm<ShapeMatchScore> algorithm = null;
        String previousEM = null;
        for (Map<String, String> c : cases) {
            ImageArray<?> em = load(root, c.get("em"));
            if (c.get("route").equals("direct")) algorithm = direct(em);
            else if (!c.get("em").equals(previousEM)) {
                algorithm = ColorDepthSearchAlgorithmProviderFactory.createShapeMatchCDSAlgorithmProvider(
                        true, null, ImageTestUtils.getExcludedRegions())
                        .createColorDepthQuerySearchAlgorithmWithDefaultParams(em, 20, 0);
            }
            previousEM = c.get("route").equals("provider") ? c.get("em") : null;
            ImageArray<?> lm = load(root, c.get("lm"));
            ImageArray<?> grad = load(root, c.get("gradient"));
            ImageTransformation clear = ImageTransformation.clearRegion(ImageTestUtils.getExcludedRegions().getRegion(lm));
            ImageArray<?> zgap = c.get("zgap") == null
                    ? ImageProcessing.create(clear).applyColorTransformation(ColorTransformation.mask(20))
                        .unsafeMaxFilter(10).applyTo(LImageUtils.create(lm)).toImageArray()
                    : load(root, c.get("zgap"));
            Map<ComputeFileType, Supplier<ImageArray<?>>> suppliers = new EnumMap<>(ComputeFileType.class);
            suppliers.put(ComputeFileType.GradientImage, () -> grad);
            suppliers.put(ComputeFileType.ZGapImage, () -> zgap);
            ShapeMatchScore s = algorithm.calculateMatchingScore(lm, suppliers);
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("case", c.get("id")); row.put("combination", c.get("combination"));
            row.put("route", c.get("route"));
            row.put("gradient_area_gap", s.getGradientAreaGap());
            row.put("high_expression_area", s.getHighExpressionArea());
            row.put("combined_long", GradientAreaGapUtils.calculate2DShapeScore(s.getGradientAreaGap(), s.getHighExpressionArea()));
            row.put("score_int", s.getScore()); row.put("mirrored", s.isMirrored());
            row.put("zgap_mode", c.get("zgap") == null ? "test-derived" : "supplied");
            row.put("zgap_rgb_sha256", pixelHash(zgap, false));
            rows.add(row);
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("evidence", "observed original Java execution");
        result.put("java_version", System.getProperty("java.runtime.version"));
        result.put("java_vendor", System.getProperty("java.vendor"));
        result.put("os", System.getProperty("os.name")); result.put("arch", System.getProperty("os.arch"));
        result.put("assets", ASSETS); result.put("real_cases", rows);
        result.put("probes", GradientProbes.capture());
        // Hashes must survive Git's LF checkout policy on every platform.
        JSON.writer(new DefaultPrettyPrinter().withObjectIndenter(new DefaultIndenter("  ", "\n")))
                .writeValue(new File(args[2]), result);
    }
}
