#!/bin/sh
# carnd-simulator-macos.zip is >100 MB so to play ball with GitHub's limitation, the following command was used to split the archive:
# split -b 64m carnd-simulator-macos.zip "carnd-simulator-macos.zip."

# Join the split archives carnd-simulator-macos.zip.aa and carnd-simulator-macos.zip.ab
cat carnd-simulator-macos.zip.* > carnd-simulator-macos.zip

# Extract temporary zip
unzip carnd-simulator-macos.zip

# Rename the ugly file name of .app simulator to something decent
mv "Default Mac desktop Universal.app" carnd-simulator-macos.app

# Delete temporary zip (optional)
rm carnd-simulator-macos.zip

# # Move simulator (optional)
# mkdir -p ~/Applications
# mv carnd-simulator-macos.app ~/Applications/.
# # Change directory
# cd ~/Applications/.

# Run the simulator
open carnd-simulator-macos.app

# # Undo change directory
# cd -