export interface PortalFilter {
  name: string;
  apply(canvas: HTMLCanvasElement, source: HTMLCanvasElement): void;
}

const dotTiles = new Map<string, HTMLCanvasElement>();

function drawSource(canvas: HTMLCanvasElement, source: HTMLCanvasElement, filter: string) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.filter = filter;
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  context.filter = "none";
}

function makeDotPattern(context: CanvasRenderingContext2D, color: string, cell = 7) {
  const key = `${color}:${cell}`;
  let tile = dotTiles.get(key);
  if (!tile) {
    tile = document.createElement("canvas");
    tile.width = cell;
    tile.height = cell;
    const tileContext = tile.getContext("2d");
    if (!tileContext) return null;
    tileContext.fillStyle = color;
    tileContext.beginPath();
    tileContext.arc(cell / 2, cell / 2, cell * 0.19, 0, Math.PI * 2);
    tileContext.fill();
    dotTiles.set(key, tile);
  }
  return context.createPattern(tile, "repeat");
}

function pixelate(canvas: HTMLCanvasElement, source: HTMLCanvasElement, transform: (data: Uint8ClampedArray, width: number, height: number) => void) {
  const sourceContext = source.getContext("2d", { willReadFrequently: true });
  const outputContext = canvas.getContext("2d", { willReadFrequently: true });
  if (!sourceContext || !outputContext) return;
  const image = sourceContext.getImageData(0, 0, canvas.width, canvas.height);
  transform(image.data, canvas.width, canvas.height);
  outputContext.putImageData(image, 0, 0);
}

function redHalftone(canvas: HTMLCanvasElement, source: HTMLCanvasElement) {
  drawSource(canvas, source, "grayscale(1) contrast(1.5) brightness(1.12)");
  const context = canvas.getContext("2d");
  if (!context) return;
  context.save();
  context.globalCompositeOperation = "multiply";
  context.fillStyle = "#ff416d";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.globalCompositeOperation = "screen";
  context.globalAlpha = 0.53;
  const pattern = makeDotPattern(context, "#ffb7cb", 7);
  if (pattern) {
    context.fillStyle = pattern;
    context.fillRect(0, 0, canvas.width, canvas.height);
  }
  context.restore();
}

function perspectiveGrid(canvas: HTMLCanvasElement, source: HTMLCanvasElement) {
  drawSource(canvas, source, "saturate(.48) brightness(.64) contrast(1.12)");
  const context = canvas.getContext("2d");
  if (!context) return;
  context.fillStyle = "rgba(114, 217, 255, .09)";
  context.fillRect(0, 0, canvas.width, canvas.height);
}

function popArt(canvas: HTMLCanvasElement, source: HTMLCanvasElement) {
  drawSource(canvas, source, "grayscale(1) contrast(1.5)");
  pixelate(canvas, canvas, (data) => {
    for (let index = 0; index < data.length; index += 4) {
      const tone = data[index];
      if (tone < 72) {
        data[index] = 13; data[index + 1] = 8; data[index + 2] = 18;
      } else if (tone < 148) {
        data[index] = 255; data[index + 1] = 27; data[index + 2] = 137;
      } else {
        data[index] = 255; data[index + 1] = 140; data[index + 2] = 25;
      }
    }
  });
}

function orderedDither(canvas: HTMLCanvasElement, source: HTMLCanvasElement) {
  drawSource(canvas, source, "grayscale(1) contrast(1.8)");
  const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
  pixelate(canvas, canvas, (data, width) => {
    for (let offset = 0, pixel = 0; offset < data.length; offset += 4, pixel++) {
      const x = pixel % width;
      const y = Math.floor(pixel / width);
      const threshold = (bayer[(y & 3) * 4 + (x & 3)] + 0.5) * 16;
      const value = data[offset] > threshold ? 242 : 13;
      data[offset] = value; data[offset + 1] = value; data[offset + 2] = value;
    }
  });
}

function chromaticShift(canvas: HTMLCanvasElement, source: HTMLCanvasElement) {
  pixelate(canvas, source, (data, width, height) => {
    const original = new Uint8ClampedArray(data);
    const shift = 3;
    for (let y = 0; y < height; y++) {
      const glitch = y % 43 < 2 ? 5 : 0;
      for (let x = 0; x < width; x++) {
        const pixel = (y * width + x) * 4;
        const redX = Math.min(width - 1, x + shift + glitch);
        const blueX = Math.max(0, x - shift - glitch);
        data[pixel] = original[(y * width + redX) * 4];
        data[pixel + 1] = original[pixel + 1];
        data[pixel + 2] = original[(y * width + blueX) * 4 + 2];
      }
    }
  });
}

function thermal(canvas: HTMLCanvasElement, source: HTMLCanvasElement) {
  drawSource(canvas, source, "grayscale(1) contrast(1.3)");
  const context = canvas.getContext("2d");
  if (!context) return;
  const gradient = context.createLinearGradient(0, canvas.height, canvas.width * 0.2, 0);
  gradient.addColorStop(0, "#13007c");
  gradient.addColorStop(0.25, "#094bff");
  gradient.addColorStop(0.48, "#00dfb7");
  gradient.addColorStop(0.7, "#ffe500");
  gradient.addColorStop(0.86, "#ff611c");
  gradient.addColorStop(1, "#fff4df");
  context.save();
  context.globalCompositeOperation = "color";
  context.fillStyle = gradient;
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.restore();
}

function vintage(canvas: HTMLCanvasElement, source: HTMLCanvasElement) {
  drawSource(canvas, source, "sepia(.8) saturate(.82) contrast(1.08)");
  const context = canvas.getContext("2d");
  if (!context) return;
  const vignette = context.createRadialGradient(
    canvas.width / 2, canvas.height / 2, canvas.height * 0.12,
    canvas.width / 2, canvas.height / 2, Math.max(canvas.width, canvas.height) * 0.74,
  );
  vignette.addColorStop(0, "rgba(35, 16, 0, 0)");
  vignette.addColorStop(1, "rgba(35, 16, 0, .62)");
  context.fillStyle = vignette;
  context.fillRect(0, 0, canvas.width, canvas.height);
}

function frostedGlass(canvas: HTMLCanvasElement, source: HTMLCanvasElement) {
  drawSource(canvas, source, "blur(9px) saturate(.48) brightness(1.18)");
  const context = canvas.getContext("2d");
  if (!context) return;
  context.fillStyle = "rgba(234, 245, 255, .4)";
  context.fillRect(0, 0, canvas.width, canvas.height);
}

export const FILTROS: PortalFilter[] = [
  { name: "Red halftone", apply: redHalftone },
  { name: "Perspective grid", apply: perspectiveGrid },
  { name: "Pop-art posterize", apply: popArt },
  { name: "B&W ordered dither", apply: orderedDither },
  { name: "Chromatic aberration", apply: chromaticShift },
  { name: "Thermal heatmap", apply: thermal },
  { name: "Sepia vintage", apply: vintage },
  { name: "Frosted glass", apply: frostedGlass },
];
