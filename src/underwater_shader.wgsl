// underwater_shader.wgsl

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var sampler: sampler;
@group(0) @binding(2) var time: f32; // For animation or distortion
@group(0) @binding(3) var color_filter: vec4<f32>; // Underwater color filter

@stage(fragment)
fn fragment_main(@location(0) frag_uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Create distortion based on time (simulating water ripples)
    let distortion_strength = 0.02;
    let offset = vec2<f32>(
        sin(frag_uv.y * 10.0 + time) * distortion_strength,
        cos(frag_uv.x * 10.0 + time) * distortion_strength
    );
    
    // Apply distortion to UV coordinates
    let distorted_uv = frag_uv + offset;
    
    // Sample the texture using distorted UVs
    let tex_color = textureSample(texture, sampler, distorted_uv);
    
    // Apply an underwater color filter (a subtle blue/green tint)
    let underwater_color = tex_color * color_filter;
    
    return underwater_color;
}

