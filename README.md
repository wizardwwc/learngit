C:\Users\Admin>code -h
Visual Studio Code 1.94.2

Usage: code.exe [options][paths...]

To read output from another program, append '-' (e.g. 'echo Hello World | code.exe -')

Options
  -d --diff <file> <file>                    Compare two files with each
                                             other.
  -m --merge <path1> <path2> <base> <result> Perform a three-way merge by
                                             providing paths for two modified
                                             versions of a file, the common
                                             origin of both modified versions
                                             and the output file to save merge
                                             results.
  -a --add <folder>                          Add folder(s) to the last active
                                             window.
  -g --goto <file:line[:character]>          Open a file at the path on the
                                             specified line and character
                                             position.
  -n --new-window                            Force to open a new window.
  -r --reuse-window                          Force to open a file or folder in
                                             an already opened window.
  -w --wait                                  Wait for the files to be closed
                                             before returning.
  --locale <locale>                          The locale to use (e.g. en-US or
                                             zh-TW).
  --user-data-dir <dir>                      Specifies the directory that user
                                             data is kept in. Can be used to
                                             open multiple distinct instances
                                             of Code.
  --profile <profileName>                    Opens the provided folder or
                                             workspace with the given profile
                                             and associates the profile with
                                             the workspace. If the profile
                                             does not exist, a new empty one
                                             is created.
  -h --help                                  Print usage.

Extensions Management
  --extensions-dir <dir>              Set the root path for extensions.
  --list-extensions                   List the installed extensions.
  --show-versions                     Show versions of installed extensions,
                                      when using --list-extensions.
  --category <category>               Filters installed extensions by provided
                                      category, when using --list-extensions.
  --install-extension <ext-id | path> Installs or updates an extension. The
                                      argument is either an extension id or a
                                      path to a VSIX. The identifier of an
                                      extension is '${publisher}.${name}'. Use
                                      '--force' argument to update to latest
                                      version. To install a specific version
                                      provide '@${version}'. For example:
                                      'vscode.csharp@1.2.3'.
  --pre-release                       Installs the pre-release version of the
                                      extension, when using
                                      --install-extension
  --uninstall-extension <ext-id>      Uninstalls an extension.
  --update-extensions                 Update the installed extensions.
  --enable-proposed-api <ext-id>      Enables proposed API features for
                                      extensions. Can receive one or more
                                      extension IDs to enable individually.

Troubleshooting
  -v --version                    Print version.
  --verbose                       Print verbose output (implies --wait).
  --log <level>                   Log level to use. Default is 'info'. Allowed
                                  values are 'critical', 'error', 'warn',
                                  'info', 'debug', 'trace', 'off'. You can
                                  also configure the log level of an extension
                                  by passing extension id and log level in the
                                  following format:
                                  '${publisher}.${name}:${logLevel}'. For
                                  example: 'vscode.csharp:trace'. Can receive
                                  one or more such entries.
  -s --status                     Print process usage and diagnostics
                                  information.
  --prof-startup                  Run CPU profiler during startup.
  --disable-extensions            Disable all installed extensions. This option
                                  is not persisted and is effective only when
                                  the command opens a new window.
  --disable-extension <ext-id>    Disable the provided extension. This option
                                  is not persisted and is effective only when
                                  the command opens a new window.
  --sync <on | off>               Turn sync on or off.
  --inspect-extensions <port>     Allow debugging and profiling of extensions.
                                  Check the developer tools for the connection
                                  URI.
  --inspect-brk-extensions <port> Allow debugging and profiling of extensions
                                  with the extension host being paused after
                                  start. Check the developer tools for the
                                  connection URI.
  --disable-lcd-text              Disable LCD font rendering.
  --disable-gpu                   Disable GPU hardware acceleration.
  --disable-chromium-sandbox      Use this option only when there is
                                  requirement to launch the application as
                                  sudo user on Linux or when running as an
                                  elevated user in an applocker environment on
                                  Windows.
  --telemetry                     Shows all telemetry events which VS code
                                  collects.

Subcommands
  tunnel       Make the current machine accessible from vscode.dev or other
               machines through a secure tunnel
  serve-web    Run a server that displays the editor UI in browsers.
