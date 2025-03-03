struct VertexInput
{
    float3 position : POSITION;
};

struct VertexOutput
{
    float4 position : SV_Position;
};

struct CameraUniform {
    float4x4 view_proj;
}

[[vk::binding(0)]]
uniform CameraUniform camera;

[shader("vertex")]
VertexOutput vertex_main(VertexInput vert)
{
    VertexOutput output;
    output.position = mul(camera.view_proj, float4(vert.position, 1.0));
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input)
{
    return float4(1.0, 1.0, 1.0, 1.0);
}
