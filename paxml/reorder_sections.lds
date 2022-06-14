/* This linker script modifies the binary layout to reduce pressure on
 * the x86-64 small code model, which requires many offsets between parts of
 * the binary to fit into a signed 32-bit integer.  Naturally, the largest such
 * offsets have the order of magnitude of the total code and data bytes in the
 * binary; but the section order also matters a good deal.
 */
SECTIONS { protodesc_cold : {} } INSERT BEFORE .rodata;
SECTIONS {
  /* Because cc_embed_data puts the object carrying the embedded blobs into
   * genfiles (whereas object files normally go into bin), and the convention of
   * naming this file <something>_data.o is so widely observed, we can use a
   * file pattern to pick out embeddings without any more explicit marking by
   * the toolchain.
   */
  .embeddings : { */genfiles/*_data.o(.rodata) }
} INSERT BEFORE protodesc_cold;
SECTIONS { .eh_frame_hdr : {} } INSERT AFTER .bss;
SECTIONS { .eh_frame : {} } INSERT AFTER .eh_frame_hdr;
SECTIONS { .gcc_except_table : {} } INSERT AFTER .eh_frame;
