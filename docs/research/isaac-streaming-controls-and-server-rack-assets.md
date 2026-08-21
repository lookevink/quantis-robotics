# Isaac Sim streaming controls and data-center assets

Research date: 2026-08-16. Deployment updated 2026-08-20 to Isaac Sim 6.0.1; the 5.0 links below preserve the APIs against which this research was originally checked.

## Short answer

Yes, the WebRTC client can forward interactive input to the remote Isaac Sim UI. NVIDIA's Kit streaming stack supports keyboard, mouse, and gamepad input, and the stock viewer is suitable for navigating the viewport, selecting objects, using menus, typing in the Script Editor, and triggering any application hotkeys or controller bindings already configured in the stage.

This is not, by itself, a robot-control protocol. WebRTC delivers input events to Kit; it does not automatically turn `W`, a mouse movement, or a gamepad axis into Franka joint or end-effector commands. A Python extension, controller/Action Graph, or ROS 2 subscriber must map those events or messages into articulation actions.

Primary sources:

- [Isaac Sim livestream clients](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/manual_livestream_clients.html)
- [Kit WebRTC extension overview](https://docs.omniverse.nvidia.com/kit/docs/omni.kit.livestream.webrtc/latest/Overview.html)
- [NVIDIA Web Viewer sample](https://github.com/NVIDIA-Omniverse/web-viewer-sample)

The Kit documentation explicitly lists keyboard, mouse, and gamepad input handling. The Web Viewer sample explains that its video element forwards keyboard and mouse events for viewport camera movement, selection, and other interactions. Client support and application bindings still determine which individual device and button combinations do useful work.

## Recommended control path for this demo

Use WebRTC for the screen and human input, then run the controller in the same Isaac Sim process:

1. Connect the desktop WebRTC client and click inside the streamed viewport so it has input focus.
2. Load the Franka scene and start simulation playback.
3. For the first proof, use the streamed Script Editor to issue a few known joint targets. Isaac Sim 5.0's basic robot tutorial demonstrates `set_joint_positions`, and its Articulation Controller accepts position, velocity, or effort commands through Python or OmniGraph.
4. For actual teleoperation, add a small Kit Python extension that subscribes to keyboard/gamepad events, converts them to end-effector deltas, runs inverse kinematics, and applies an `ArticulationAction` on every physics callback. Log the resulting explicit action vector alongside observations; do not reconstruct training actions later from key presses.

Relevant APIs and examples:

- [Basic robot control](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/introduction/quickstart_isaacsim_robot.html)
- [Articulation Controller: Python and OmniGraph](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/robot_simulation/articulation_controller.html)
- [Isaac Sim extension and standalone workflows](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/introduction/workflows.html)
- [OmniGraph controller shortcuts](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/omnigraph/omnigraph_shortcuts.html)

The controller shortcuts can create joint position/velocity, differential-drive, and gripper graphs. NVIDIA documents built-in WASD generation for differential drive and `O`/`C`/`N` bindings for a gripper. They do not provide a complete six-degree-of-freedom Franka keyboard teleoperator, so arm motion still needs a mapping and IK/controller layer.

### When to use ROS 2 instead

Use the ROS 2 Bridge once control is coming from a separate policy process, an existing robot stack, or a future real robot. A typical graph subscribes to joint commands and connects the subscriber to an Articulation Controller; the bridge can also publish joint and sensor state. ROS publishers/subscribers are active only while simulation playback is running.

- [ROS 2 joint-command controller example](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/ros2_tutorials/tutorial_ros2_rl_controller.html)
- [ROS 2 bridge overview](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/ros2_tutorials/ros2_landing_page.html)
- [ROS 2 simulation control](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/ros2_tutorials/tutorial_ros2_simulation_control.html)

ROS 2 gives a clean boundary and reusable message contracts, but DDS discovery/networking across a cloud VM is extra work. Prefer a private VPN or carefully scoped network configuration rather than opening DDS broadly. For a one-person demo, an in-process Python extension is the shortest path.

### Custom controls over WebRTC

NVIDIA's Web Viewer sample also supports bidirectional custom JSON messages with `AppStreamer.sendMessage()`. A custom Kit extension must receive the event and apply the robot command. This is useful for building browser buttons or a joystick UI, but it requires a custom web viewer; the stock desktop client is best treated as remote GUI/input, not a general robot RPC client.

## Server-rack and data-center assets

### Best source: NVIDIA Data Center Assets Pack

NVIDIA publishes a purpose-built **Data Center Assets Pack** containing 33 OpenUSD models: servers, server racks, network switches, NVIDIA UFM, PDUs, patch panels, blanking panels, and cable trays. It is 9.8 GB and is the best starting point for this demo because it is already OpenUSD rather than a CAD mesh that still needs conversion.

- [NVIDIA downloadable OpenUSD asset packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html#data-center-assets-pack)
- [Direct Data Center Assets Pack download](https://d4i3qtqj3r0z5.cloudfront.net/Datacenter_NVD%4010012.zip)

The download endpoint was reachable at the research date. Download and unzip the complete pack on the EC2 instance's EBS volume, then preserve its directory structure: USD files commonly reference sibling geometry, textures, and MDL materials by relative path.

### Assets already in the Isaac Sim 5.0 catalog

The default Isaac 5.0 online catalog contains generic rack props, but none is explicitly named as a server cabinet. Treat these as quick blockout assets, not as authoritative data-center equipment:

- [`/Isaac/Environments/Office/Props/SM_Rack.usd`](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Office/Props/SM_Rack.usd)
- [`/Isaac/Environments/Office/Props/SM_Rack1m.usd`](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Office/Props/SM_Rack1m.usd)
- [`/Isaac/Environments/Office/Props/SM_RackA.usd`](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Office/Props/SM_RackA.usd)
- [`/Isaac/Environments/Simple_Warehouse/Props/SM_RackFrame_03.usd`](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props/SM_RackFrame_03.usd) is warehouse/pallet shelving, not a server rack.
- [`warehouse_multiple_shelves.usd`](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd) is useful as an immediate room/layout stand-in.

Browse these from **Window > Browsers > Isaac Sim Assets** or the Content Browser. NVIDIA documents the 5.0 catalog roots and downloadable local asset packs here:

- [Isaac Sim Assets overview](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/assets/usd_assets_overview.html)
- [Isaac Sim Asset Browser](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/py/source/extensions/isaacsim.asset.browser/docs/index.html)
- [Isaac Sim 5.0 local asset packs](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_faq.html#isaac-sim-setup-tips)

### Manufacturer CAD/BIM for higher fidelity

For the eventual partner-specific demo, ask the known data-center equipment manufacturer for a simplified CAD/BIM model and explicit permission to use it for simulation, generated training data, screenshots/video, and partner demos. Public first-party sources include:

- [Schneider Electric/APC NetShelter racks](https://www.se.com/us/en/work/products/master-ranges/netshelter/)
- [Vertiv VR Rack, including BIM/CAD downloads](https://www.vertiv.com/en-us/products-catalog/facilities-enclosures-and-racks/racks-and-containment/vertiv-vr-rack/)
- [Legrand BIM model downloads](https://www.legrand.us/resources/bim%20models)
- [Siemon BIM Download Center](https://www.siemon.com/en/support/bim/)

NVIDIA documents that Isaac Sim can convert OBJ, FBX, and glTF through the Asset Importer and supports common CAD sources through its CAD Converter/connectors. After import, verify meters/scale, normals and materials; simplify excess CAD detail; add collision geometry and rigid-body properties; and split doors, handles, removable server modules, and cables into the prims/articulations required by the task.

- [Isaac Sim 5.0 importing assets](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/introduction/reference_architecture.html#importing-assets)
- [Asset-to-USD conversion script](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/python_scripting/environment_setup.html#convert-asset-to-usd)
- [Asset Validator](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/robot_setup/asset_validation.html)

## Licensing and security notes

NVIDIA's asset-pack page says the downloadable assets are free to use in projects. Use remains governed by NVIDIA's software/product terms and any notices included in the pack; preserve its README/license files and verify redistribution rights before committing the 9.8 GB pack to GitHub or shipping the raw models to others. Manufacturer CAD/BIM is governed by the manufacturer's download terms; public download does not automatically grant redistribution or ML-dataset rights.

- [Current Omniverse licensing summary](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/NVIDIA_Omniverse_License_Agreement.html)
- [Isaac Sim additional software and materials license](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/common/license-isaac-sim-additional.html)

Finally, current NVIDIA livestream documentation warns that streaming endpoints are intended for private/trusted networks and should not be exposed without safeguards. Keep TCP 49100 and UDP 47998 restricted to the operator's IP or a VPN; only one streaming client can connect to an Isaac Sim instance at a time.
