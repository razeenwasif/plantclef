# PlantCLEF 2026 manim renders.
# Q=l|m|h|k → 480p15 | 720p30 | 1080p60 | 4K60. Defaults to h.
Q ?= h

# Frame sampling rate for `make frames` (frames per second to extract).
# 0.4 fps → one frame every 2.5 s → ~24 frames out of a 60 s scene.
FPS ?= 0.4

# Pass-through to manim. Each target renders ONE scene from its file.
MANIM := manim -q$(Q) --disable_caching

# Quality token used in manim's output path tree.
QDIR_l = 480p15
QDIR_m = 720p30
QDIR_h = 1080p60
QDIR_k = 2160p60
QDIR   = $(QDIR_$(Q))

.PHONY: all pipeline logit aggregation phenology blender-hero combined clean frames

all: pipeline logit aggregation phenology

pipeline:
	$(MANIM) scene_pipeline_overview.py PipelineOverview

logit:
	$(MANIM) scene_logit_adjustment.py LogitAdjustment

aggregation:
	$(MANIM) scene_bayesian_aggregation.py BayesianAggregation

phenology:
	$(MANIM) scene_phenology.py PhenologyPenalty

# 3D establishing shot rendered with Blender 4.x (Eevee Next). See
# blender/README.md for prerequisites and integration steps.
blender-hero:
	@command -v blender >/dev/null 2>&1 || { \
	    echo "blender not on PATH; install with: sudo apt install blender"; \
	    exit 1; \
	}
	cd blender && blender --background --python hero_quadrat.py -- --render

# Stitch the Blender hero MP4 onto the front of the unified Manim
# pipeline scene (which now includes the phenology deep dive) and write
# one final media/combined.mp4.
#
# We unconditionally re-encode both inputs into a single 1080p60 H.264
# stream. The earlier stream-copy path was technically faster but
# preserved each input's own codec timebase, which the concat demuxer
# would mux into a single MP4 whose reported framerate then read as
# something jittery and low (the player's "effective" fps dropped as
# soon as the demuxer hit the Blender → Manim seam). Re-encoding is a
# few seconds longer and produces a clean, uniformly-paced output.
#
# If the Blender hero is missing (Blender not installed, or Cycles
# refused to render headless), the target degrades gracefully and just
# copies the Manim render to media/combined.mp4 with a warning.
combined: pipeline
	@$(MAKE) -s blender-hero || echo "[combined] blender-hero failed or skipped — continuing without hero shot"
	@mkdir -p media
	@MANIM_MP4="$$PWD/media/videos/scene_pipeline_overview/$(QDIR)/PipelineOverview.mp4"; \
	BLENDER_MP4=$$(ls "$$PWD/blender/renders/hero_quadrat"*.mp4 2>/dev/null | tail -n 1); \
	if [ ! -f "$$MANIM_MP4" ]; then \
	    echo "[combined] ERROR: Manim render missing: $$MANIM_MP4"; \
	    exit 1; \
	fi; \
	if [ ! -f "$$BLENDER_MP4" ]; then \
	    echo "[combined] Blender hero MP4 not found — copying Manim render alone"; \
	    cp "$$MANIM_MP4" media/combined.mp4; \
	    echo "[combined] media/combined.mp4 (Manim only · pipeline + phenology)"; \
	    exit 0; \
	fi; \
	ffmpeg -hide_banner -loglevel error \
	    -i "$$BLENDER_MP4" -i "$$MANIM_MP4" \
	    -filter_complex "[0:v]fps=60,setsar=1[v0];[1:v]fps=60,setsar=1[v1];[v0][v1]concat=n=2:v=1:a=0[outv]" \
	    -map "[outv]" \
	    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
	    -movflags +faststart \
	    -y media/combined.mp4; \
	echo "[combined] media/combined.mp4 (1080p60 CFR · hero + pipeline + phenology)"

# Extract evenly-spaced keyframes from every rendered scene into
# frames/<scene>/000.png … so they can be reviewed without playing
# the video. Re-run `make frames` after each render iteration.
frames:
	@mkdir -p frames
	@for scene in PipelineOverview LogitAdjustment BayesianAggregation PhenologyPenalty; do \
	    case $$scene in \
	        PipelineOverview)     dir=scene_pipeline_overview ;; \
	        LogitAdjustment)      dir=scene_logit_adjustment ;; \
	        BayesianAggregation)  dir=scene_bayesian_aggregation ;; \
	        PhenologyPenalty)     dir=scene_phenology ;; \
	    esac; \
	    src=media/videos/$$dir/$(QDIR)/$$scene.mp4; \
	    if [ -f $$src ]; then \
	        echo "[frames] $$scene"; \
	        rm -rf frames/$$scene; mkdir -p frames/$$scene; \
	        ffmpeg -loglevel error -y -i $$src -vf "fps=$(FPS)" \
	               frames/$$scene/%03d.png; \
	    else \
	        echo "[frames] SKIP $$scene (no $$src — run \`make $$(echo $$scene | tr A-Z a-z | head -c 9)\` first)"; \
	    fi; \
	done
	@echo "[frames] done — see frames/<scene>/*.png"

clean:
	rm -rf media/ frames/
