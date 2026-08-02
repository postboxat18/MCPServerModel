from mcp.server.fastmcp import FastMCP

from mcp.server.transport_security import TransportSecuritySettings

mcp = FastMCP(
    "calculator_server",
    json_response=True,
    host="0.0.0.0",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["mcpserver-fzg9.onrender.com", "mcpserver-fzg9.onrender.com:*"],
        allowed_origins=["https://mcpserver-fzg9.onrender.com"],
    ),
)


@mcp.tool()
def add(a: int, b: int):
    print("Calling add")
    return a + b


@mcp.tool()
def sub(a: int, b: int):
    print("Calling sub")
    return a - b


@mcp.tool()
def divide(a: int, b: int):
    print("Calling div")
    return a // b


@mcp.tool()
def mul(a: int, b: int):
    print("Calling mul")
    return a * b


transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
app = mcp.streamable_http_app()
