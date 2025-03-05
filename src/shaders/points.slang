struct VertexInput
{
    float3 position;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float2 diff;
};

struct CameraUniform
{
    float4x4 view_proj;
};

struct WorldUniform
{
    float2 resolution;
};

[vk_binding(0, 0)]
ConstantBuffer<CameraUniform> camera;

[vk_binding(1, 0)]
ConstantBuffer<WorldUniform> world;

[shader("vertex")]
VertexOutput vertex_main(VertexInput vert, int vertexId: SV_VertexID)
{
    let points = float2[](
        float2(-1.0, 1.0),
        float2(-1.0, -1.0),
        float2(1.0, 1.0),
        float2(1.0, -1.0), );
    VertexOutput output;
    let pos = points[vertexId];

    output.position = mul(camera.view_proj, float4(vert.position, 1.0));
    output.position.xy += pos * 0.01 / world.resolution * world.resolution.y;
    output.diff = pos;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input)
{
    float2 uv = abs(input.diff);
    float d = sqrt(dot(uv, uv));
    if (d > 1)
        discard;

    float3 norm = float3(input.diff.x, input.diff.y, sqrt(1.0 - dot(input.diff, input.diff)));

    float3 c = float3(1.5, 0.75, 0.5) * input.position.z;\

    c += 0.5;
    c *= norm.z * 0.5 + 0.5;
    
    return float4(c, 1.0);
}
