use hdf5_metno_derive::H5Type;

#[derive(H5Type)]
//~^ ERROR proc-macro derive
//~^^ HELP H5Type can only be derived for enums with explicit representation
enum Foo {
    X = 1,
    Y = 2,
}

fn main() {}
