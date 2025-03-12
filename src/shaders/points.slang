public struct CameraUniform
{
    public float4x4 view;
    public float4x4 proj;
    public float4x4 view_proj;
};

public struct WorldUniform
{
    public float2 resolution;
};
struct VertexInput
{
    float3 position : POSITION;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float2 diff;
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
    let d = input.diff;
    let p = dot(d, d);
    if (p > 1)
    {
        discard;
    }
    float3 c = float3(3.0, 1.5, 1.0);

    // for normal shading
    // float3 norm = float3(input.diff.x, input.diff.y, sqrt(1.0 - dot(input.diff, input.diff)));
    // norm = normalize(norm);

    // float3 light_dir = normalize(mul(camera.view, float4(0.0, 20.0, 0.0, 1.0)).xyz);
    // float light = max(dot(norm, light_dir.xyz), 0.0) * 0.5 + 0.5;
    // c.xyz *= light;

    c.xyz *= input.position.z;
    c.xyz *= 0.5;
    c.xyz += 0.5;

    return float4(c, 1.0);
}
