import ij.IJ;
import ij.Menus;

/** Test-only launcher: all colour encoding/projection runs in upstream classes. */
public final class RunReference {
    public static void main(String[] args) {
        Menus.getCommands();
        String result = IJ.runMacroFile(args[0]);
        if (result != null && result.startsWith("[aborted]")) {
            System.err.println(result);
            System.exit(1);
        }
        System.exit(0);
    }
}
