import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, Sphere, MeshDistortMaterial } from '@react-three/drei';
import * as THREE from 'three';
import { THEME_COLORS } from '../lib/site';

const PulsingNucleus = ({ status }: { status: string }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const color = status === 'training' ? THEME_COLORS.vitalsBall.training : THEME_COLORS.vitalsBall.idle;

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = state.clock.getElapsedTime() * 0.2;
      meshRef.current.rotation.y = state.clock.getElapsedTime() * 0.3;
    }
  });

  return (
    <Float speed={2} rotationIntensity={1} floatIntensity={1}>
      <Sphere ref={meshRef} args={[1, 100, 100]} scale={1.5}>
        <MeshDistortMaterial
          color={color}
          speed={3}
          distort={0.4}
          radius={1}
          emissive={color}
          emissiveIntensity={0.5}
          roughness={0.2}
          metalness={0.8}
        />
      </Sphere>
      {/* Outer shell */}
      <Sphere args={[1.8, 64, 64]}>
        <meshStandardMaterial
          color={color}
          wireframe
          transparent
          opacity={0.1}
        />
      </Sphere>
    </Float>
  );
};

export const CoreVitals3D = ({ status }: { status: string }) => {
  return (
    <div className="w-full h-full min-h-[300px] relative">
      <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#fff" />
        <spotLight position={[-10, -10, -10]} intensity={0.5} color={THEME_COLORS.vitalsSpot} />
        <PulsingNucleus status={status} />
      </Canvas>
      <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
        <div className="w-48 h-48 rounded-full border border-oracle-accent/20 animate-glow-spin" />
        <div className="absolute w-64 h-64 rounded-full border border-oracle-cyan/10 animate-pulse-slow" />
      </div>
    </div>
  );
};
