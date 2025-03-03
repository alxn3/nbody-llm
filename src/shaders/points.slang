struct VertexInput
{
    float3 position : POSITION;
};

struct VertexOutput
{
    float4 position : SV_Position;
};

struct CameraUniform
{
    float4x4 view_proj;
};

struct WorldUniform
{
    float2 resolution;
};


[vk_binding(0,0)]
ConstantBuffer<CameraUniform> camera;

[vk_binding(1,0)]
ConstantBuffer<WorldUniform> world;

[shader("vertex")]
VertexOutput vertex_main(VertexInput vert, int vertexId: SV_VertexID)
{ 
    let points = float2[](
        float2(-1.0, 1.0),
        float2(-1.0, -1.0),
        float2(1.0, 1.0),
        float2(1.0, -1.0),
    );
    VertexOutput output;
    let pos = points[vertexId];

    output.position = mul(camera.view_proj, float4(vert.position, 1.0));
    output.position.xy += pos * min(0.05 / output.position.z, 0.25) / world.resolution * min(world.resolution.x, world.resolution.y);
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input)
{
    return float4(1.0, 1.0, 1.0, 1.0);
}
