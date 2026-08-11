import { app } from "../../scripts/app.js";

const PUBLIC = new Set([
    "MiniMaxH3LatentLabLongMediaSetup",
    "MiniMaxH3LatentLabLongMediaSampler",
    "MiniMaxH3LatentLabLongMediaDecode",
]);
const CATEGORY = "MiniMax H3/Long Media";

app.registerExtension({
    name: "MiniMaxH3LatentLab.NodeFacade.v2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cls = nodeType?.comfyClass ?? nodeType?.ComfyClass ?? nodeData?.name;
        if (!cls?.startsWith?.("MiniMaxH3LatentLab")) return;

        if (PUBLIC.has(cls)) {
            nodeType.skip_list = false;
            nodeType.category = CATEGORY;
            if (nodeData) {
                nodeData.hidden = false;
                nodeData.category = CATEGORY;
            }
            return;
        }

        // Keep internal classes registered for GraphBuilder/legacy workflows.
        nodeType.skip_list = true;
        if (nodeData) nodeData.hidden = true;
    },
});
