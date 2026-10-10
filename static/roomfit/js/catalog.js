// Typical furniture sizes (cm) to start from. Adjust to the real product's spec sheet.
// shape: rect | ellipse | L (corner sofa: lw = chaise width, ld = seat depth of main part)

export const CATALOG = [
  { group: 'Living', name: 'Sofa 3-seat', w: 210, d: 90, h: 85, color: '#7c8ea3' },
  { group: 'Living', name: 'Sofa 2-seat', w: 160, d: 88, h: 85, color: '#7c8ea3' },
  { group: 'Living', name: 'Corner sofa', shape: 'L', w: 260, d: 160, h: 85, lw: 90, ld: 90, color: '#7c8ea3' },
  { group: 'Living', name: 'Armchair', w: 80, d: 85, h: 95, color: '#9aa7b8' },
  { group: 'Living', name: 'Coffee table', w: 110, d: 60, h: 42, color: '#b08b62' },
  { group: 'Living', name: 'Round coffee table', shape: 'ellipse', w: 80, d: 80, h: 42, color: '#b08b62' },
  { group: 'Living', name: 'TV bench', w: 180, d: 40, h: 50, color: '#5d5d66' },
  { group: 'Living', name: 'Bookshelf', w: 80, d: 30, h: 200, color: '#c9a77c' },
  { group: 'Living', name: 'Rug', w: 200, d: 140, h: 1, color: '#d9c7a7' },

  { group: 'Bedroom', name: 'Bed 90×200', w: 98, d: 210, h: 45, color: '#a7b6a0' },
  { group: 'Bedroom', name: 'Bed 140×200', w: 150, d: 210, h: 45, color: '#a7b6a0' },
  { group: 'Bedroom', name: 'Bed 160×200', w: 170, d: 212, h: 45, color: '#a7b6a0' },
  { group: 'Bedroom', name: 'Bed 180×200', w: 190, d: 212, h: 45, color: '#a7b6a0' },
  { group: 'Bedroom', name: 'Nightstand', w: 45, d: 40, h: 55, color: '#c9a77c' },
  { group: 'Bedroom', name: 'Wardrobe 2-door', w: 100, d: 60, h: 200, color: '#e2ddd3' },
  { group: 'Bedroom', name: 'Wardrobe 3-door', w: 150, d: 60, h: 236, color: '#e2ddd3' },
  { group: 'Bedroom', name: 'Dresser', w: 80, d: 48, h: 80, color: '#c9a77c' },

  { group: 'Dining & work', name: 'Dining table 4p', w: 140, d: 80, h: 75, color: '#b08b62' },
  { group: 'Dining & work', name: 'Dining table 6p', w: 180, d: 90, h: 75, color: '#b08b62' },
  { group: 'Dining & work', name: 'Round table Ø110', shape: 'ellipse', w: 110, d: 110, h: 75, color: '#b08b62' },
  { group: 'Dining & work', name: 'Chair', w: 45, d: 52, h: 85, color: '#8a7360' },
  { group: 'Dining & work', name: 'Desk', w: 140, d: 70, h: 75, color: '#e2ddd3' },
  { group: 'Dining & work', name: 'Office chair', shape: 'ellipse', w: 65, d: 65, h: 110, color: '#444a55' },

  { group: 'Kitchen & bath', name: 'Fridge', w: 60, d: 65, h: 185, color: '#dfe3e8' },
  { group: 'Kitchen & bath', name: 'Washing machine', w: 60, d: 60, h: 85, color: '#dfe3e8' },
  { group: 'Kitchen & bath', name: 'Kitchen unit 60', w: 60, d: 60, h: 90, color: '#e2ddd3' },
  { group: 'Kitchen & bath', name: 'Wall cabinet 60', w: 60, d: 35, h: 70, elevation: 145, color: '#e2ddd3' },

  { group: 'Other', name: 'Plant', shape: 'ellipse', w: 45, d: 45, h: 120, color: '#6f9a6a' },
  { group: 'Other', name: 'Custom box', w: 100, d: 50, h: 100, color: '#b9a0c9' },
];
