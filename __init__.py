from .nodes import AmageTextNode, AmageOneNode, AmageFpsConverterNode, AmageSTTNode

NODE_CLASS_MAPPINGS = {
    "Amage Text": AmageTextNode,
    "Amage All in One": AmageOneNode,
    "Amage FPS Converter": AmageFpsConverterNode,
    "Amage STT": AmageSTTNode,
}
