package org.janelia.colormipsearch.cds;

import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.ShortProcessor;
import org.janelia.colormipsearch.ImageTestUtils;
import org.janelia.colormipsearch.imageprocessing.ColorTransformation;
import org.janelia.colormipsearch.imageprocessing.ImageArray;
import org.janelia.colormipsearch.imageprocessing.ImageArrayUtils;
import org.janelia.colormipsearch.imageprocessing.ImageTransformation;
import org.janelia.colormipsearch.imageprocessing.LImage;
import org.janelia.colormipsearch.imageprocessing.LImageUtils;
import org.janelia.colormipsearch.model.ComputeFileType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.lang.reflect.Method;
import java.util.function.Supplier;

/** Test-only observations from the pinned colormipsearch implementation. */
public final class GradientProbes {
    private GradientProbes() { }

    public static Map<String, Object> capture() {
        Map<String, Object> result = new LinkedHashMap<String, Object>();

        List<Map<String, Object>> scalar = new ArrayList<Map<String, Object>>();
        int[][] colors = {
                {0, 0, 0}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
                {255, 255, 0}, {255, 0, 255}, {0, 255, 255},
                {127, 127, 127}, {128, 128, 128}, {254, 254, 254},
                {255, 255, 255}, {127, 0, 255}, {0, 254, 255},
                {255, 252, 3}, {255, 0, 200}
        };
        for (int[] a : colors) {
            for (int[] b : colors) {
                Map<String, Object> row = new LinkedHashMap<String, Object>();
                int rgbA = rgb(a), rgbB = rgb(b);
                row.put("rgbA", rgbA);
                row.put("rgbB", rgbB);
                row.put("sliceGap", GradientAreaGapUtils.calculateSliceGap(rgbA, rgbB));
                scalar.add(row);
            }
        }
        result.put("sliceGapColorPairs", scalar);

        result.put("sectorLutProbes", sectorLutProbes());
        List<Map<String, Object>> boundary = sliceBoundaryProbes();
        result.put("sliceGap79to81", boundary);

        List<Map<String, Object>> arithmetic = new ArrayList<Map<String, Object>>();
        long[][] pairs = {{79, 0}, {80, 0}, {81, 0}, {0, 2}, {0, 3}, {0, 4},
                {2147483647L, 0}, {2147483648L, 0}, {Long.MAX_VALUE, 1}, {Long.MAX_VALUE, 3}, {0, Long.MAX_VALUE}};
        for (long[] p : pairs) {
            Map<String, Object> row = new LinkedHashMap<String, Object>();
            row.put("gap", p[0]); row.put("highExpression", p[1]);
            row.put("combinedLong", GradientAreaGapUtils.calculate2DShapeScore(p[0], p[1]));
            row.put("combinedInt", (int) GradientAreaGapUtils.calculate2DShapeScore(p[0], p[1]));
            arithmetic.add(row);
        }
        for (Long[] p : new Long[][]{{null, 0L}, {0L, null}, {-1L, 0L}, {0L, -1L}}) {
            Map<String, Object> row = new LinkedHashMap<String, Object>();
            row.put("gap", p[0]); row.put("highExpression", p[1]);
            row.put("combinedLong", GradientAreaGapUtils.calculate2DShapeScore(p[0], p[1]));
            arithmetic.add(row);
        }
        result.put("combinedLong", arithmetic);
        Map<String, Object> narrowing = new LinkedHashMap<String, Object>();
        ShapeMatchScore longScore = new ShapeMatchScore(2147483648L, 0L, -1L, false);
        narrowing.put("sourceLong", longScore.getGradientAreaGap());
        narrowing.put("combinedLong", GradientAreaGapUtils.calculate2DShapeScore(longScore.getGradientAreaGap(), longScore.getHighExpressionArea()));
        narrowing.put("scoreInt", longScore.getScore());
        result.put("shapeScoreIntNarrowing", narrowing);

        List<Map<String, Object>> normalized = new ArrayList<Map<String, Object>>();
        long[][] normCases = {{0, 0, 0, 0}, {0, 1, 5, 10}, {5, 0, 9, 0}, {1, 1, 0, 1}, {1, 1, 1, 0},
                {1, 0, 1, 1}, {1, 100, 1, 100}, {1, 1, 1, 1},
                {100, 1, 1, 1}, {-1, 1, 1, 1}, {1, Long.MAX_VALUE, 1, 1}};
        for (long[] p : normCases) {
            double value = GradientAreaGapUtils.calculateNormalizedScore((int) p[0], p[1], p[2], p[3]);
            Map<String, Object> row = new LinkedHashMap<String, Object>();
            row.put("pixelMatch", p[0]); row.put("shapeScore", p[1]);
            row.put("maxPixelMatch", p[2]); row.put("maxShapeScore", p[3]);
            putNormalized(row, value);
            normalized.add(row);
        }
        result.put("normalization", normalized);
        result.put("upstreamNormalizationRows", upstreamNormalizationRows());
        result.put("normalizationCandidates", normalizationCandidates());

        List<Map<String, Object>> grays = new ArrayList<Map<String, Object>>();
        for (int rgb : new int[]{0x000000, 0x000001, 0x000101, 0x000201, 0x010000, 0x010101,
                0x7F0000, 0x7F7F7F, 0x808080, 0xFEFEFE, 0xFFFFFF, 0x010203}) {
            Map<String, Object> row = new LinkedHashMap<String, Object>();
            row.put("rgb", rgb);
            row.put("gray16NoGamma", LImageUtils.create(colorArray(1, 1, new int[]{rgb}))
                    .map(ColorTransformation.toGray16WithNoGammaCorrection()).fold(0, (pixel, sum) -> pixel));
            grays.add(row);
        }
        result.put("grayRounding", grays);

        result.put("unsafeMaxFilter", filterProbes());

        Map<String, Object> labels = new LinkedHashMap<String, Object>();
        for (int width : new int[]{270, 271, 600, 601}) {
            ImageArray<?> blank = colorArray(width, 101, new int[width * 101]);
            java.util.function.BiPredicate<Integer, Integer> excluded = ImageTestUtils.getExcludedRegions().getRegion(blank);
            Map<String, Object> points = new LinkedHashMap<String, Object>();
            for (int[] xy : new int[][]{{0, 0}, {329, 99}, {330, 0}, {330, 89}, {330, 90}, {329, 100},
                    {width - 271, 89}, {width - 270, 89}, {width - 1, 89}, {width - 1, 90}}) {
                points.put(xy[0] + "," + xy[1], excluded.test(xy[0], xy[1]));
            }
            labels.put(String.valueOf(width), points);
        }
        result.put("excludedRegion", labels);

        List<Map<String, Object>> scoring = new ArrayList<Map<String, Object>>();
        scoring.add(scoreCase("single-pixel-contribution-3", new int[]{0x00FF0000}, new int[]{3}, new int[]{0}, false));
        scoring.add(scoreCase("single-pixel-contribution-4", new int[]{0x00FF0000}, new int[]{4}, new int[]{0}, false));
        scoring.add(scoreCase("unmirrored-tie", new int[]{0x00FF0000}, new int[]{0}, new int[]{0}, true));
        scoring.add(roiCase());
        scoring.add(mirroredGapCandidates());
        scoring.add(noShiftCase());
        scoring.add(missingVariantCase());
        result.put("syntheticScorer", scoring);

        return result;
    }

    private static Map<String, Object> scoreCase(String name, int[] query, int[] gradient, int[] zGap, boolean mirror) {
        int w = query.length;
        LImage queryImage = LImageUtils.create(colorArray(w, 1, query));
        LImage mask = LImageUtils.create(byteArray(w, 1, filled(w, 1)));
        LImage high = LImageUtils.create(byteArray(w, 1, new int[w]));
        Shape2DMatchColorDepthSearchAlgorithm algorithm = new Shape2DMatchColorDepthSearchAlgorithm(
                queryImage, mask, high, null, 20, mirror, ImageTransformation.IDENTITY);
        ImageArray<?> target = colorArray(w, 1, new int[w]);
        ImageArray<?> grad = gray16Array(w, 1, gradient);
        ImageArray<?> zgapImage = colorArray(w, 1, zGap);
        Map<ComputeFileType, Supplier<ImageArray<?>>> suppliers = new LinkedHashMap<ComputeFileType, Supplier<ImageArray<?>>>();
        suppliers.put(ComputeFileType.GradientImage, () -> grad);
        suppliers.put(ComputeFileType.ZGapImage, () -> zgapImage);
        ShapeMatchScore score = algorithm.calculateMatchingScore(target, suppliers);
        Map<String, Object> row = new LinkedHashMap<String, Object>();
        row.put("case", name); row.put("queryPixels", query); row.put("gradientPixels", gradient);
        row.put("zGapPixels", zGap); row.put("mirrorQuery", mirror);
        row.put("queryInputTransform", "identity");
        row.put("gradientAreaGap", score.getGradientAreaGap());
        row.put("highExpressionArea", score.getHighExpressionArea());
        row.put("combinedScore", score.getScore()); row.put("mirrored", score.isMirrored());
        return row;
    }

    private static Map<String, Object> roiCase() {
        int[] query = {0x00FF0000, 0x0000FF00};
        LImage queryImage = LImageUtils.create(colorArray(2, 1, query));
        LImage mask = LImageUtils.create(byteArray(2, 1, new int[]{1, 1}));
        LImage high = LImageUtils.create(byteArray(2, 1, new int[]{0, 0}));
        LImage roi = LImageUtils.create(colorArray(2, 1, new int[]{0x00FFFFFF, 0}));
        ImageArray<?> target = colorArray(2, 1, new int[2]);
        ImageArray<?> gradient = gray16Array(2, 1, new int[]{4, 8});
        ImageArray<?> zgap = byteArray(2, 1, new int[2]);
        Map<String, Object> row = new LinkedHashMap<String, Object>();
        row.put("case", "fixed-coordinate-roi"); row.put("queryPixels", query);
        row.put("roiPixels", new int[]{0x00FFFFFF, 0}); row.put("gradientPixels", new int[]{4, 8});
        row.put("unmirroredOnly", calculateScore(queryImage, mask, high, roi, target, gradient, zgap, false));
        row.put("mirrorEnabled", calculateScore(queryImage, mask, high, roi, target, gradient, zgap, true));
        row.put("individualTrials", scoreTrials(queryImage, mask, high, roi, LImageUtils.create(target),
                LImageUtils.create(gradient), LImageUtils.create(zgap)));
        return row;
    }

    private static Map<String, Object> calculateScore(LImage query, LImage mask, LImage high, LImage roi,
                                                        ImageArray<?> target, ImageArray<?> gradient,
                                                        ImageArray<?> zgap, boolean mirror) {
        Shape2DMatchColorDepthSearchAlgorithm algorithm = new Shape2DMatchColorDepthSearchAlgorithm(
                query, mask, high, roi, 20, mirror, ImageTransformation.IDENTITY);
        Map<ComputeFileType, Supplier<ImageArray<?>>> suppliers = new LinkedHashMap<ComputeFileType, Supplier<ImageArray<?>>>();
        suppliers.put(ComputeFileType.GradientImage, () -> gradient);
        suppliers.put(ComputeFileType.ZGapImage, () -> zgap);
        ShapeMatchScore score = algorithm.calculateMatchingScore(target, suppliers);
        Map<String, Object> result = new LinkedHashMap<String, Object>();
        result.put("gradientAreaGap", score.getGradientAreaGap()); result.put("highExpressionArea", score.getHighExpressionArea());
        result.put("combinedScore", score.getScore()); result.put("mirrored", score.isMirrored());
        return result;
    }

    private static Map<String, Object> mirroredGapCandidates() {
        int[] queryPixels = {0x00FF0000, 0x0000FF00, 0x000000FF};
        LImage query = LImageUtils.create(colorArray(3, 1, queryPixels));
        LImage mask = LImageUtils.create(byteArray(3, 1, new int[]{1, 1, 1}));
        LImage high = LImageUtils.create(byteArray(3, 1, new int[]{0, 0, 0}));
        LImage target = LImageUtils.create(colorArray(3, 1, new int[3]));
        LImage gradient = LImageUtils.create(gray16Array(3, 1, new int[]{4, 8, 16}));
        int[] zGapPixels = {0x0000FFFF, 0x00FF0000, 0};
        LImage zgap = LImageUtils.create(colorArray(3, 1, zGapPixels));
        Map<String, Object> row = new LinkedHashMap<String, Object>();
        row.put("case", "asymmetric-mirrored-z-gap-trials"); row.put("queryPixels", queryPixels);
        row.put("gradientPixels", new int[]{4, 8, 16}); row.put("zGapPixels", zGapPixels);
        row.put("trials", scoreTrials(query, mask, high, null, target, gradient, zgap));
        // A deliberate supplier perturbation cancels the scorer's Z-gap mirror.
        // Observe it through original Java too; do not calculate that score here.
        row.put("supplierMirroredZGapPixels", new int[]{0, 0x00FF0000, 0x0000FFFF});
        row.put("supplierMirroredTrials", scoreTrials(query, mask, high, null, target, gradient,
                zgap.mapi(ImageTransformation.horizontalMirror())));
        return row;
    }

    private static Map<String, Object> scoreTrials(LImage query, LImage mask, LImage high, LImage roi,
                                                   LImage target, LImage gradient, LImage zgap) {
        Map<String, Object> row = new LinkedHashMap<String, Object>();
        try {
            Method calculate = Shape2DMatchColorDepthSearchAlgorithm.class.getDeclaredMethod(
                    "calculateNegativeScores", LImage.class, LImage.class, LImage.class, ImageTransformation.class, boolean.class);
            calculate.setAccessible(true);
            Shape2DMatchColorDepthSearchAlgorithm algorithm = new Shape2DMatchColorDepthSearchAlgorithm(
                    query, mask, high, roi, 20, true, ImageTransformation.IDENTITY);
            for (boolean mirrored : new boolean[]{false, true}) {
                ShapeMatchScore score = (ShapeMatchScore) calculate.invoke(algorithm, target, gradient, zgap,
                        mirrored ? ImageTransformation.horizontalMirror() : ImageTransformation.IDENTITY, mirrored);
                Map<String, Object> trial = new LinkedHashMap<String, Object>();
                trial.put("mirroredInput", mirrored); trial.put("gradientAreaGap", score.getGradientAreaGap());
                trial.put("highExpressionArea", score.getHighExpressionArea()); trial.put("combinedScore", score.getScore());
                trial.put("mirrored", score.isMirrored()); row.put(mirrored ? "mirroredTrial" : "unmirroredTrial", trial);
            }
        } catch (Exception e) {
            throw new IllegalStateException("Could not observe original mirrored score trials", e);
        }
        return row;
    }

    private static Map<String, Object> noShiftCase() {
        int[] queryPixels = {0, 0x00FF0000, 0};
        LImage query = LImageUtils.create(colorArray(3, 1, queryPixels));
        LImage mask = LImageUtils.create(byteArray(3, 1, new int[]{0, 1, 0}));
        LImage high = LImageUtils.create(byteArray(3, 1, new int[3]));
        ImageArray<?> target = colorArray(3, 1, new int[3]);
        ImageArray<?> zGap = byteArray(3, 1, new int[3]);
        Map<String, Object> row = new LinkedHashMap<String, Object>();
        row.put("case", "no-positive-shift-search"); row.put("queryPixels", queryPixels);
        row.put("queryMask", new int[]{0, 1, 0}); row.put("alignedGradient", new int[]{0, 4, 0});
        row.put("onePixelShiftedGradient", new int[]{0, 0, 4}); row.put("queryInputTransform", "identity");
        row.put("alignedOutput", calculateScore(query, mask, high, null, target, gray16Array(3, 1, new int[]{0, 4, 0}), zGap, false));
        row.put("shiftedOutput", calculateScore(query, mask, high, null, target, gray16Array(3, 1, new int[]{0, 0, 4}), zGap, false));
        return row;
    }

    private static Map<String, Object> missingVariantCase() {
        LImage query = LImageUtils.create(colorArray(1, 1, new int[]{0x00FF0000}));
        LImage mask = LImageUtils.create(byteArray(1, 1, new int[]{1}));
        LImage high = LImageUtils.create(byteArray(1, 1, new int[]{0}));
        Shape2DMatchColorDepthSearchAlgorithm algorithm = new Shape2DMatchColorDepthSearchAlgorithm(
                query, mask, high, null, 20, false, ImageTransformation.IDENTITY);
        ImageArray<?> target = colorArray(1, 1, new int[1]);
        Map<String, Object> row = new LinkedHashMap<String, Object>();
        row.put("case", "missing-variant-sentinel");
        for (ComputeFileType omitted : new ComputeFileType[]{ComputeFileType.GradientImage, ComputeFileType.ZGapImage}) {
            Map<ComputeFileType, Supplier<ImageArray<?>>> suppliers = new LinkedHashMap<ComputeFileType, Supplier<ImageArray<?>>>();
            for (ComputeFileType type : new ComputeFileType[]{ComputeFileType.GradientImage, ComputeFileType.ZGapImage}) {
                if (type != omitted) suppliers.put(type, () -> colorArray(1, 1, new int[]{0}));
            }
            ShapeMatchScore score = algorithm.calculateMatchingScore(target, suppliers);
            Map<String, Object> outcome = new LinkedHashMap<String, Object>();
            outcome.put("omittedSupplier", omitted.toString()); outcome.put("gradientAreaGap", score.getGradientAreaGap());
            outcome.put("highExpressionArea", score.getHighExpressionArea()); outcome.put("combinedScore", score.getScore());
            row.put("without" + omitted.toString(), outcome);
        }
        return row;
    }

    private static List<Map<String, Object>> sectorLutProbes() {
        List<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        try {
            Class<?> colorType = Class.forName(GradientAreaGapUtils.class.getName() + "$Color");
            Method method = GradientAreaGapUtils.class.getDeclaredMethod("findSliceNumber", colorType, colorType, double.class);
            method.setAccessible(true);
            Object[][] sectors = {
                    {enumValue(colorType, "RED"), enumValue(colorType, "GREEN")},
                    {enumValue(colorType, "RED"), enumValue(colorType, "BLUE")},
                    {enumValue(colorType, "GREEN"), enumValue(colorType, "RED")},
                    {enumValue(colorType, "GREEN"), enumValue(colorType, "BLUE")},
                    {enumValue(colorType, "BLUE"), enumValue(colorType, "RED")},
                    {enumValue(colorType, "BLUE"), enumValue(colorType, "GREEN")}
            };
            double[] ratios = {0d, 1d / 255d, 0.5d, 254d / 255d, 1d};
            for (Object[] sector : sectors) for (double ratio : ratios) {
                Map<String, Object> row = new LinkedHashMap<String, Object>();
                row.put("maximumChannel", sector[0].toString());
                row.put("secondChannel", sector[1].toString());
                row.put("ratio", ratio);
                row.put("sliceNumber", method.invoke(null, sector[0], sector[1], ratio));
                rows.add(row);
            }
            Method lutMethod = GradientAreaGapUtils.class.getDeclaredMethod("findSliceNumberInLUT", int.class, int.class, double.class);
            lutMethod.setAccessible(true);
            double tieRatio = (90d + 93d) / (2d * 255d);
            Map<String, Object> tie = new LinkedHashMap<String, Object>();
            tie.put("sector", "BLUE_GREEN"); tie.put("lutRangeStart", 30); tie.put("lutRangeEnd", 85);
            tie.put("midpointRatio", tieRatio); tie.put("observedSliceNumber", lutMethod.invoke(null, 30, 85, tieRatio));
            rows.add(tie);
        } catch (Exception e) {
            throw new IllegalStateException("Could not observe original sector lookup", e);
        }
        return rows;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static Object enumValue(Class<?> type, String name) {
        return Enum.valueOf((Class<? extends Enum>) type, name);
    }

    private static List<Map<String, Object>> sliceBoundaryProbes() {
        List<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        int query = 0x00FF0000;
        int[] wanted = {79, 80, 81};
        int[] selected = new int[wanted.length];
        int[][] colorFunctions = {{1, 0}, {2, 0}, {0, 1}, {0, 2}, {1, 2}, {2, 1}};
        for (int[] channels : colorFunctions) {
            for (int level = 0; level <= 255; level++) {
                int[] color = new int[3];
                color[channels[0]] = 255;
                color[channels[1]] = level;
                int candidate = rgb(color);
                int observed = GradientAreaGapUtils.calculateSliceGap(query, candidate);
                for (int i = 0; i < wanted.length; i++) if (selected[i] == 0 && observed == wanted[i]) selected[i] = candidate;
            }
        }
        for (int i = 0; i < wanted.length; i++) {
            if (selected[i] == 0) throw new IllegalStateException("Original slice lookup produced no pair for requested gap " + wanted[i]);
            Map<String, Object> row = new LinkedHashMap<String, Object>();
            row.put("queryRgb", query); row.put("targetDilatedRgb", selected[i]);
            row.put("observedSliceGap", GradientAreaGapUtils.calculateSliceGap(query, selected[i]));
            row.put("scorerObservation", scoreCase("slice-gap-" + wanted[i], new int[]{query},
                    new int[]{wanted[i] == 79 ? 4 : 0}, new int[]{selected[i]}, false));
            rows.add(row);
        }
        return rows;
    }

    private static List<Map<String, Object>> normalizationCandidates() {
        List<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        Map<String, Object> cohort = new LinkedHashMap<String, Object>();
        long[] pixelCandidates = {9, 2}, shapeCandidates = {4, 13};
        cohort.put("candidatePixelScores", pixelCandidates); cohort.put("candidateShapeScores", shapeCandidates);
        cohort.put("independentMaximumPixel", Math.max(pixelCandidates[0], pixelCandidates[1]));
        cohort.put("independentMaximumShape", Math.max(shapeCandidates[0], shapeCandidates[1]));
        cohort.put("candidateA", new long[]{9, 4}); cohort.put("candidateB", new long[]{2, 13});
        long pMax = Math.max(pixelCandidates[0], pixelCandidates[1]);
        long nMax = Math.max(shapeCandidates[0], shapeCandidates[1]);
        cohort.put("candidateANormalized", GradientAreaGapUtils.calculateNormalizedScore((int)pixelCandidates[0], shapeCandidates[0], pMax, nMax));
        cohort.put("candidateBNormalized", GradientAreaGapUtils.calculateNormalizedScore((int)pixelCandidates[1], shapeCandidates[1], pMax, nMax));
        rows.add(cohort);
        for (int base = 100000000; base < 100001000; base++) {
            double a = GradientAreaGapUtils.calculateNormalizedScore(1, base, 1, 1000000000L);
            double b = GradientAreaGapUtils.calculateNormalizedScore(1, base + 1, 1, 1000000000L);
            if (a != b && (float) a == (float) b) {
                Map<String, Object> collision = new LinkedHashMap<String, Object>();
                collision.put("inputsA", new long[]{1, base, 1, 1000000000L});
                collision.put("inputsB", new long[]{1, base + 1L, 1, 1000000000L});
                collision.put("utilityDoubleA", a); collision.put("utilityDoubleB", b);
                org.janelia.colormipsearch.model.CDMatchEntity<?, ?> entityA = new org.janelia.colormipsearch.model.CDMatchEntity<>();
                org.janelia.colormipsearch.model.CDMatchEntity<?, ?> entityB = new org.janelia.colormipsearch.model.CDMatchEntity<>();
                entityA.updateNormalizedScore((float) a); entityB.updateNormalizedScore((float) b);
                collision.put("storedFloatA", entityA.getNormalizedScore());
                collision.put("storedFloatB", entityB.getNormalizedScore());
                collision.put("storedFloatBitsA", Float.floatToIntBits(entityA.getNormalizedScore()));
                collision.put("storedFloatBitsB", Float.floatToIntBits(entityB.getNormalizedScore()));
                rows.add(collision);
                break;
            }
        }
        if (rows.size() < 2) throw new IllegalStateException("Original normalization utility produced no distinct-double float collision");
        return rows;
    }

    private static List<Map<String, Object>> upstreamNormalizationRows() {
        List<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        long[][] cases = {{636, 156, 1897, 679, 1114361}, {636, 233, 1644, 679, 1107088},
                {636, 0, 1644, 679, 1114361}, {795, 123, 93, 875, 1606182}};
        for (long[] c : cases) {
            long shape = GradientAreaGapUtils.calculate2DShapeScore(c[1], c[2]);
            double normalized = GradientAreaGapUtils.calculateNormalizedScore((int) c[0], shape, c[3], c[4]);
            Map<String, Object> row = new LinkedHashMap<String, Object>();
            row.put("pixelMatch", c[0]); row.put("gap", c[1]); row.put("highExpression", c[2]);
            row.put("maxPixelMatch", c[3]);
            row.put("maxShapeScore", c[4]);
            row.put("shapeScore", shape); putNormalized(row, normalized); rows.add(row);
        }
        Map<String, Object> sentinel = new LinkedHashMap<String, Object>();
        sentinel.put("pixelMatch", 636); sentinel.put("shapeScore", -1L);
        sentinel.put("maxPixelMatch", 679); sentinel.put("maxShapeScore", 1114361L);
        sentinel.put("normalizedDouble", GradientAreaGapUtils.calculateNormalizedScore(636, -1, 679, 1114361L));
        rows.add(sentinel);
        return rows;
    }

    private static List<Map<String, Object>> filterProbes() {
        List<Map<String, Object>> rows = new ArrayList<Map<String, Object>>();
        int[] corner = pixelsWithOne(25, 0, 0, 0x00FFFFFF);
        int[] centerChannels = new int[25]; centerChannels[12] = 0x00FF0000; centerChannels[13] = 0x0000FF00; centerChannels[17] = 0x000000FF;
        for (String location : new String[]{"corner", "center-multichannel"}) for (int border : new int[]{0, 1}) for (double radius : new double[]{1d, 1.5d, 1.75d, 2.5d, 2.85d}) {
            int[] input = location.equals("corner") ? corner : centerChannels;
            LImage source = LImageUtils.create(colorArray(5, 5, input), border, border, border, border);
            LImage filtered = source.mapi(ImageTransformation.unsafeMaxFilter(radius));
            ImageArray<?> materialized = filtered.toImageArray();
            List<Integer> output = imagePixels(materialized);
            List<Integer> mirrored = imagePixels(filtered.mapi(ImageTransformation.horizontalMirror()).toImageArray());
            Map<String, Object> row = new LinkedHashMap<String, Object>();
            row.put("location", location); row.put("width", 5); row.put("height", 5);
            row.put("borderPixels", border); row.put("radius", radius);
            row.put("inputPixels", input); row.put("materializedPixels", output); row.put("mirroredPixels", mirrored);
            row.put("materializedSum", sumPixels(materialized));
            row.put("firstTraversalSum", filtered.fold(0L, (p, s) -> s + p));
            row.put("secondTraversalSum", filtered.fold(0L, (p, s) -> s + p));
            row.put("reusedPixels", imagePixels(filtered.toImageArray()));
            LImage reduced = filtered.reduce();
            row.put("reducedPixels", imagePixels(reduced.toImageArray()));
            row.put("reducedRepeatedPixels", imagePixels(reduced.toImageArray()));
            row.put("freshPixels", imagePixels(filterImage(input, border, radius).toImageArray()));
            row.put("freshMirroredPixels", imagePixels(filterImage(input, border, radius)
                    .mapi(ImageTransformation.horizontalMirror()).toImageArray()));
            row.put("evaluationOrder", Arrays.asList("materialize lazy", "mirror reused lazy", "fold reused lazy",
                    "fold reused lazy again", "materialize reused lazy", "reduce reused lazy",
                    "materialize reduced twice", "materialize fresh lazy", "mirror fresh lazy"));
            rows.add(row);
        }
        return rows;
    }

    private static List<Integer> imagePixels(ImageArray<?> image) {
        List<Integer> pixels = new ArrayList<Integer>();
        for (int i = 0; i < image.getPixelCount(); i++) pixels.add(image.get(i));
        return pixels;
    }

    private static LImage filterImage(int[] input, int border, double radius) {
        return LImageUtils.create(colorArray(5, 5, input), border, border, border, border)
                .mapi(ImageTransformation.unsafeMaxFilter(radius));
    }

    private static void putNormalized(Map<String, Object> row, double value) {
        org.janelia.colormipsearch.model.CDMatchEntity<?, ?> entity = new org.janelia.colormipsearch.model.CDMatchEntity<>();
        entity.updateNormalizedScore((float)value);
        row.put("normalizedDouble", value);
        row.put("storedFloat32", entity.getNormalizedScore());
        row.put("storedFloat32Bits", Float.floatToIntBits(entity.getNormalizedScore()));
        row.put("storedFloat32Exact", (double)entity.getNormalizedScore());
    }

    private static long sumPixels(ImageArray<?> image) {
        long sum = 0;
        for (int i = 0; i < image.getPixelCount(); i++) sum += image.get(i);
        return sum;
    }

    private static ImageArray<?> byteArray(int w, int h, int[] pixels) {
        byte[] values = new byte[pixels.length];
        for (int i = 0; i < pixels.length; i++) values[i] = (byte) pixels[i];
        return ImageArrayUtils.fromImagePlus(new ImagePlus("gradient-probe-mask", new ByteProcessor(w, h, values)));
    }

    private static ImageArray<?> gray16Array(int w, int h, int[] pixels) {
        short[] values = new short[pixels.length];
        for (int i = 0; i < pixels.length; i++) values[i] = (short) pixels[i];
        return ImageArrayUtils.fromImagePlus(new ImagePlus("gradient-probe-gray16", new ShortProcessor(w, h, values, null)));
    }

    private static int rgb(int[] c) { return (c[0] << 16) | (c[1] << 8) | c[2]; }
    private static int[] filled(int n, int value) { int[] a = new int[n]; Arrays.fill(a, value); return a; }
    private static int[] pixelsWithOne(int n, int x, int y, int value) { int[] a = new int[n]; a[y * 5 + x] = value; return a; }
    private static ImageArray<?> colorArray(int w, int h, int[] pixels) {
        return ImageArrayUtils.fromImagePlus(new ImagePlus("gradient-probe", new ColorProcessor(w, h, pixels)));
    }
}
